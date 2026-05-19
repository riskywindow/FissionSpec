"""Executable CPU fidelity primitives for outcome-cached speculative serving.

This module is deliberately composable rather than a replacement for the
count-level policy simulator. It adds mechanisms that can be exercised without
a GPU: finite paged outcome caches, correlated request classes, context-aware
costs, a deterministic multi-worker remote draft service, and a one-round
prefill/verification trace.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, TypeAlias

from .profiles import HardwareProfile
from .rng import CounterRNG, Seed

CacheKey: TypeAlias = tuple[str, int, int]
QueuePolicy = Literal["continuous-batching", "priority"]


def _finite_non_negative(value: float, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0.0
    ):
        raise ValueError(f"{field} must be finite and non-negative")
    return float(value)


def _positive_integer(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _probability(value: float, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{field} must be finite and in [0, 1]")
    return float(value)


@dataclass(frozen=True, slots=True)
class OutcomeClass:
    """Latent request class coupling acceptance and outcome popularity."""

    class_id: str
    acceptance_probability: float
    outcome_probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.class_id, str) or not self.class_id:
            raise ValueError("class_id must be a non-empty string")
        _probability(self.acceptance_probability, field="acceptance_probability")
        if not self.outcome_probabilities:
            raise ValueError("outcome_probabilities must not be empty")
        for probability in self.outcome_probabilities:
            _probability(probability, field="outcome_probability")
        if not math.isclose(
            math.fsum(self.outcome_probabilities),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("outcome_probabilities must sum to one")

    def popular_outcomes(self, fanout: int) -> tuple[int, ...]:
        """Return top outcomes with deterministic outcome-id tie breaking."""

        _positive_integer(fanout, field="fanout")
        ordered = sorted(
            range(len(self.outcome_probabilities)),
            key=lambda outcome: (-self.outcome_probabilities[outcome], outcome),
        )
        return tuple(ordered[:fanout])


@dataclass(frozen=True, slots=True)
class FidelityRequest:
    """One heterogeneous request entering the CPU fidelity trace."""

    request_id: str
    arrival_ms: float
    prompt_tokens: int
    output_tokens: int
    speculation_length: int
    class_weights: tuple[tuple[str, float], ...]
    correlation_key: str | None = None
    priority: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be a non-empty string")
        _finite_non_negative(self.arrival_ms, field="arrival_ms")
        if (
            isinstance(self.prompt_tokens, bool)
            or not isinstance(self.prompt_tokens, int)
            or self.prompt_tokens < 0
        ):
            raise ValueError("prompt_tokens must be a non-negative integer")
        _positive_integer(self.output_tokens, field="output_tokens")
        _positive_integer(self.speculation_length, field="speculation_length")
        if not self.class_weights:
            raise ValueError("class_weights must not be empty")
        class_ids = tuple(class_id for class_id, _ in self.class_weights)
        if any(not isinstance(class_id, str) or not class_id for class_id in class_ids):
            raise ValueError("class identifiers must be non-empty strings")
        if len(class_ids) != len(set(class_ids)):
            raise ValueError("class identifiers must be unique within class_weights")
        for _, weight in self.class_weights:
            _probability(weight, field="class_weight")
        if not math.isclose(
            math.fsum(weight for _, weight in self.class_weights),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("class_weights must sum to one")
        if self.correlation_key is not None and (
            not isinstance(self.correlation_key, str) or not self.correlation_key
        ):
            raise ValueError("correlation_key must be a non-empty string when supplied")
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise ValueError("priority must be an integer")


@dataclass(frozen=True, slots=True)
class ResolvedRequest:
    """Request after deterministic latent-class resolution."""

    request: FidelityRequest
    outcome_class: OutcomeClass


def _categorical(probabilities: tuple[float, ...], draw: float) -> int:
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if draw < cumulative or index == len(probabilities) - 1:
            return index
    raise AssertionError("categorical distribution had no terminal interval")


def resolve_request_classes(
    requests: tuple[FidelityRequest, ...],
    classes: tuple[OutcomeClass, ...],
    *,
    seed: Seed,
) -> tuple[ResolvedRequest, ...]:
    """Resolve independent or correlation-key-shared latent classes."""

    class_by_id = {item.class_id: item for item in classes}
    if len(class_by_id) != len(classes):
        raise ValueError("outcome class identifiers must be unique")
    if not requests:
        raise ValueError("at least one request is required")
    if len({request.request_id for request in requests}) != len(requests):
        raise ValueError("request identifiers must be unique")
    rng = CounterRNG(seed)
    group_resolution: dict[str, tuple[tuple[str, float], ...]] = {}
    group_class: dict[str, str] = {}
    resolved: list[ResolvedRequest] = []
    for request in requests:
        unknown = {class_id for class_id, _ in request.class_weights} - set(class_by_id)
        if unknown:
            raise ValueError(f"request {request.request_id!r} names unknown classes: {unknown}")
        # Domain separation prevents an independent request named ``x`` from
        # accidentally sharing a draw with correlation group ``x``.
        draw_key = (
            f"group/{request.correlation_key}"
            if request.correlation_key is not None
            else f"request/{request.request_id}"
        )
        if request.correlation_key is not None:
            prior = group_resolution.setdefault(draw_key, request.class_weights)
            if prior != request.class_weights:
                raise ValueError("requests sharing correlation_key must share class_weights")
        if draw_key not in group_class:
            draw = rng.uniform(draw_key, 0, "latent-request-class")
            selected = _categorical(
                tuple(weight for _, weight in request.class_weights),
                draw,
            )
            group_class[draw_key] = request.class_weights[selected][0]
        resolved.append(
            ResolvedRequest(
                request=request,
                outcome_class=class_by_id[group_class[draw_key]],
            )
        )
    return tuple(resolved)


def sample_accepted_tokens(
    resolved: ResolvedRequest,
    *,
    round_id: int,
    seed: Seed,
) -> int:
    """Sample the accepted prefix; the target bonus token is not included."""

    if isinstance(round_id, bool) or not isinstance(round_id, int) or round_id < 0:
        raise ValueError("round_id must be a non-negative integer")
    rng = CounterRNG(seed)
    accepted = 0
    width = min(
        resolved.request.speculation_length,
        resolved.request.output_tokens,
    )
    for draw in range(width - 1):
        if (
            rng.uniform(
                resolved.request.request_id,
                round_id,
                f"acceptance/{resolved.outcome_class.class_id}",
                draw,
            )
            < resolved.outcome_class.acceptance_probability
        ):
            accepted += 1
        else:
            break
    return accepted


def sample_realized_outcome(
    resolved: ResolvedRequest,
    *,
    round_id: int,
    seed: Seed,
) -> int:
    """Sample a realized next-continuation outcome from class popularity."""

    rng = CounterRNG(seed)
    draw = rng.uniform(
        resolved.request.request_id,
        round_id,
        f"outcome/{resolved.outcome_class.class_id}",
    )
    return _categorical(resolved.outcome_class.outcome_probabilities, draw)


@dataclass(frozen=True, slots=True)
class CacheEntry:
    key: CacheKey
    logical_bytes: int
    pages: int
    allocated_bytes: int
    last_access_tick: int


@dataclass(frozen=True, slots=True)
class CacheMutation:
    admitted: bool
    key: CacheKey
    evicted_keys: tuple[CacheKey, ...]
    logical_bytes: int
    allocated_pages: int
    allocated_bytes: int


class OutcomeTreeCache:
    """Finite page-rounded cache with deterministic LRU eviction."""

    __slots__ = (
        "_allocated_pages",
        "_entries",
        "_logical_bytes",
        "_peak_allocated_pages",
        "_tick",
        "byte_budget",
        "page_budget",
        "page_size_bytes",
    )

    def __init__(self, *, byte_budget: int, page_size_bytes: int) -> None:
        _positive_integer(byte_budget, field="byte_budget")
        _positive_integer(page_size_bytes, field="page_size_bytes")
        self.byte_budget = byte_budget
        self.page_size_bytes = page_size_bytes
        self.page_budget = byte_budget // page_size_bytes
        if self.page_budget == 0:
            raise ValueError("byte_budget must hold at least one page")
        self._entries: dict[CacheKey, CacheEntry] = {}
        self._tick = 0
        self._logical_bytes = 0
        self._allocated_pages = 0
        self._peak_allocated_pages = 0

    @property
    def logical_bytes(self) -> int:
        return self._logical_bytes

    @property
    def allocated_pages(self) -> int:
        return self._allocated_pages

    @property
    def allocated_bytes(self) -> int:
        return self._allocated_pages * self.page_size_bytes

    @property
    def peak_allocated_pages(self) -> int:
        return self._peak_allocated_pages

    @property
    def keys(self) -> tuple[CacheKey, ...]:
        return tuple(sorted(self._entries))

    def contains(self, key: CacheKey) -> bool:
        """Return exact membership without changing LRU order."""

        return key in self._entries

    def discard(self, key: CacheKey) -> bool:
        """Release one resident entry without changing unrelated recency."""

        if key not in self._entries:
            return False
        self._remove(key)
        return True

    def _remove(self, key: CacheKey) -> None:
        entry = self._entries.pop(key)
        self._logical_bytes -= entry.logical_bytes
        self._allocated_pages -= entry.pages

    def insert(self, key: CacheKey, *, logical_bytes: int) -> CacheMutation:
        """Insert one branch, evicting exact LRU entries until it fits."""

        _positive_integer(logical_bytes, field="logical_bytes")
        pages = math.ceil(logical_bytes / self.page_size_bytes)
        if pages > self.page_budget:
            return CacheMutation(
                admitted=False,
                key=key,
                evicted_keys=(),
                logical_bytes=self.logical_bytes,
                allocated_pages=self.allocated_pages,
                allocated_bytes=self.allocated_bytes,
            )
        self._tick += 1
        if key in self._entries:
            self._remove(key)
        evicted: list[CacheKey] = []
        while self._allocated_pages + pages > self.page_budget:
            victim = min(
                self._entries.values(),
                key=lambda entry: (entry.last_access_tick, entry.key),
            )
            evicted.append(victim.key)
            self._remove(victim.key)
        entry = CacheEntry(
            key=key,
            logical_bytes=logical_bytes,
            pages=pages,
            allocated_bytes=pages * self.page_size_bytes,
            last_access_tick=self._tick,
        )
        self._entries[key] = entry
        self._logical_bytes += logical_bytes
        self._allocated_pages += pages
        self._peak_allocated_pages = max(
            self._peak_allocated_pages,
            self._allocated_pages,
        )
        return CacheMutation(
            admitted=True,
            key=key,
            evicted_keys=tuple(evicted),
            logical_bytes=self.logical_bytes,
            allocated_pages=self.allocated_pages,
            allocated_bytes=self.allocated_bytes,
        )

    def lookup(self, key: CacheKey) -> bool:
        """Return membership and update recency only on a hit."""

        entry = self._entries.get(key)
        if entry is None:
            return False
        self._tick += 1
        self._entries[key] = CacheEntry(
            key=entry.key,
            logical_bytes=entry.logical_bytes,
            pages=entry.pages,
            allocated_bytes=entry.allocated_bytes,
            last_access_tick=self._tick,
        )
        return True

    def audit(self) -> None:
        """Fail loudly if logical and physical accounting diverge."""

        if self._logical_bytes != sum(entry.logical_bytes for entry in self._entries.values()):
            raise AssertionError("cache logical-byte accounting diverged")
        if self._allocated_pages != sum(entry.pages for entry in self._entries.values()):
            raise AssertionError("cache page accounting diverged")
        if self.allocated_bytes > self.byte_budget:
            raise AssertionError("cache exceeded byte budget")


@dataclass(frozen=True, slots=True)
class ContextCostModel:
    """Context-dependent CPU latency and transport surface."""

    prefill_base_ms: float = 0.0
    prefill_per_token_ms: float = 0.0
    target_base_ms: float = 0.0
    target_per_row_ms: float = 0.0
    target_per_context_token_ms: float = 0.0
    target_per_verifier_slot_ms: float = 0.0
    draft_base_ms: float = 0.0
    recovery_base_ms: float = 0.0
    draft_per_row_ms: float = 0.0
    draft_per_context_token_ms: float = 0.0
    draft_per_branch_ms: float = 0.0
    network_base_ms: float = 0.0
    network_per_byte_ms: float = 0.0
    network_jitter_ms: float = 0.0
    reference_profile: HardwareProfile | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "prefill_base_ms",
            "prefill_per_token_ms",
            "target_base_ms",
            "target_per_row_ms",
            "target_per_context_token_ms",
            "target_per_verifier_slot_ms",
            "draft_base_ms",
            "recovery_base_ms",
            "draft_per_row_ms",
            "draft_per_context_token_ms",
            "draft_per_branch_ms",
            "network_base_ms",
            "network_per_byte_ms",
            "network_jitter_ms",
        ):
            _finite_non_negative(getattr(self, field_name), field=field_name)

    @classmethod
    def reference(cls, profile: HardwareProfile) -> ContextCostModel:
        """Collapse exactly to the existing row/slot hardware abstraction."""

        return cls(reference_profile=profile)

    def prefill_ms(self, prompt_tokens: int) -> float:
        if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int):
            raise ValueError("prompt_tokens must be an integer")
        if prompt_tokens < 0:
            raise ValueError("prompt_tokens must be non-negative")
        if self.reference_profile is not None:
            return 0.0
        return self.prefill_base_ms + prompt_tokens * self.prefill_per_token_ms

    def target_ms(
        self,
        *,
        context_tokens: tuple[int, ...],
        verifier_slots: int,
    ) -> float:
        rows = len(context_tokens)
        _positive_integer(rows, field="target rows")
        _positive_integer(verifier_slots, field="verifier_slots")
        if any(
            isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0
            for tokens in context_tokens
        ):
            raise ValueError("context_tokens must contain non-negative integers")
        if verifier_slots < rows:
            raise ValueError("verifier_slots must be at least target rows")
        if self.reference_profile is not None:
            return self.reference_profile.target_latency_ms(rows, verifier_slots)
        return (
            self.target_base_ms
            + rows * self.target_per_row_ms
            + math.fsum(context_tokens) * self.target_per_context_token_ms
            + verifier_slots * self.target_per_verifier_slot_ms
        )

    def remote_service_ms(
        self,
        *,
        recovery: bool,
        context_tokens: tuple[int, ...],
        branches: tuple[int, ...],
    ) -> float:
        rows = len(context_tokens)
        _positive_integer(rows, field="remote rows")
        if len(branches) != rows:
            raise ValueError("branches must contain one count per remote row")
        if any(
            isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0
            for tokens in context_tokens
        ):
            raise ValueError("context_tokens must contain non-negative integers")
        for branch_count in branches:
            _positive_integer(branch_count, field="branch count")
        if self.reference_profile is not None:
            return self.reference_profile.draft_latency_ms(rows, recovery=recovery)
        return (
            (self.recovery_base_ms if recovery else self.draft_base_ms)
            + rows * self.draft_per_row_ms
            + math.fsum(context_tokens) * self.draft_per_context_token_ms
            + math.fsum(branches) * self.draft_per_branch_ms
        )

    def network_ms(
        self,
        *,
        payload_bytes: int,
        rng: CounterRNG,
        job_id: str,
        attempt: int,
        direction: str,
    ) -> float:
        if isinstance(payload_bytes, bool) or not isinstance(payload_bytes, int):
            raise ValueError("payload_bytes must be an integer")
        if payload_bytes < 0:
            raise ValueError("payload_bytes must be non-negative")
        if self.reference_profile is not None:
            return 0.0
        jitter = (
            rng.uniform(job_id, attempt, f"network-jitter/{direction}") * self.network_jitter_ms
        )
        return self.network_base_ms + payload_bytes * self.network_per_byte_ms + jitter


class DraftJobKind(StrEnum):
    PRECOMPUTE = "precompute"
    RECOVERY = "recovery"


@dataclass(frozen=True, slots=True)
class DraftJob:
    job_id: str
    request_id: str
    kind: DraftJobKind
    submit_ms: float
    context_tokens: int
    branches: int
    payload_bytes: int
    priority: int = 0

    def __post_init__(self) -> None:
        for field_name in ("job_id", "request_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a non-empty string")
        if not isinstance(self.kind, DraftJobKind):
            raise TypeError("kind must be DraftJobKind")
        _finite_non_negative(self.submit_ms, field="submit_ms")
        if (
            isinstance(self.context_tokens, bool)
            or not isinstance(self.context_tokens, int)
            or self.context_tokens < 0
        ):
            raise ValueError("context_tokens must be a non-negative integer")
        _positive_integer(self.branches, field="branches")
        if (
            isinstance(self.payload_bytes, bool)
            or not isinstance(self.payload_bytes, int)
            or self.payload_bytes < 0
        ):
            raise ValueError("payload_bytes must be a non-negative integer")
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise ValueError("priority must be an integer")


@dataclass(frozen=True, slots=True)
class RemoteDraftConfig:
    workers: int = 1
    queue_policy: QueuePolicy = "continuous-batching"
    max_batch_size: int = 16
    batch_window_ms: float = 0.0
    queue_capacity: int = 128
    failure_probability: float = 0.0
    max_retries: int = 0
    retry_backoff_ms: float = 0.0

    def __post_init__(self) -> None:
        _positive_integer(self.workers, field="workers")
        if self.queue_policy not in {"continuous-batching", "priority"}:
            raise ValueError("queue_policy must be continuous-batching or priority")
        _positive_integer(self.max_batch_size, field="max_batch_size")
        _positive_integer(self.queue_capacity, field="queue_capacity")
        _finite_non_negative(self.batch_window_ms, field="batch_window_ms")
        _probability(self.failure_probability, field="failure_probability")
        if (
            isinstance(self.max_retries, bool)
            or not isinstance(self.max_retries, int)
            or self.max_retries < 0
        ):
            raise ValueError("max_retries must be a non-negative integer")
        _finite_non_negative(self.retry_backoff_ms, field="retry_backoff_ms")

    @classmethod
    def reference(cls, *, max_batch_size: int = 16) -> RemoteDraftConfig:
        """One reliable FIFO worker with zero batching delay or backpressure."""

        return cls(
            workers=1,
            queue_policy="continuous-batching",
            max_batch_size=max_batch_size,
            batch_window_ms=0.0,
            queue_capacity=max_batch_size,
        )


@dataclass(frozen=True, slots=True)
class DraftAttemptRecord:
    job_id: str
    request_id: str
    kind: DraftJobKind
    attempt: int
    worker_id: int
    batch_id: int
    network_ready_ms: float
    admitted_ms: float
    service_start_ms: float
    service_end_ms: float
    response_ms: float
    backpressure_delay_ms: float
    success: bool
    terminal: bool


@dataclass(frozen=True, slots=True)
class DraftBatchRecord:
    batch_id: int
    worker_id: int
    kind: DraftJobKind
    job_ids: tuple[str, ...]
    start_ms: float
    end_ms: float


@dataclass(frozen=True, slots=True)
class RemoteDraftTrace:
    attempts: tuple[DraftAttemptRecord, ...]
    batches: tuple[DraftBatchRecord, ...]
    successful_job_ids: tuple[str, ...]
    terminal_failed_job_ids: tuple[str, ...]
    worker_available_ms: tuple[float, ...]
    queue_peak: int
    retries: int
    backpressured_attempts: int

    def completion_by_job(self) -> dict[str, DraftAttemptRecord]:
        return {
            attempt.job_id: attempt
            for attempt in self.attempts
            if attempt.success or attempt.terminal
        }


@dataclass(frozen=True, slots=True)
class _Attempt:
    job: DraftJob
    attempt: int
    network_ready_ms: float


@dataclass(frozen=True, slots=True)
class _Waiting:
    attempt: _Attempt
    admitted_ms: float


@dataclass(slots=True)
class _Formation:
    kind: DraftJobKind
    deadline_ms: float
    selected: list[_Waiting]


class RemoteDraftService:
    """Deterministic non-preemptive multi-worker remote draft scheduler."""

    def __init__(
        self,
        config: RemoteDraftConfig,
        costs: ContextCostModel,
        *,
        seed: Seed,
    ) -> None:
        self.config = config
        self.costs = costs
        self.rng = CounterRNG(seed)

    @staticmethod
    def _attempt_key(attempt: _Attempt) -> tuple[float, str, int]:
        return (attempt.network_ready_ms, attempt.job.job_id, attempt.attempt)

    def _waiting_key(self, waiting: _Waiting) -> tuple[int, float, str, int]:
        attempt = waiting.attempt
        return (
            -attempt.job.priority if self.config.queue_policy == "priority" else 0,
            attempt.network_ready_ms,
            attempt.job.job_id,
            attempt.attempt,
        )

    def run(
        self,
        jobs: tuple[DraftJob, ...],
        *,
        initial_worker_available_ms: tuple[float, ...] | None = None,
    ) -> RemoteDraftTrace:
        """Run all jobs through transport, queueing, batching, failure, and retry."""

        if not jobs:
            raise ValueError("remote draft service needs at least one job")
        if len({job.job_id for job in jobs}) != len(jobs):
            raise ValueError("draft job identifiers must be unique")
        worker_available = (
            [0.0] * self.config.workers
            if initial_worker_available_ms is None
            else list(initial_worker_available_ms)
        )
        if len(worker_available) != self.config.workers or any(
            not math.isfinite(value) or value < 0.0 for value in worker_available
        ):
            raise ValueError("initial_worker_available_ms must match workers and be non-negative")
        source: list[_Attempt] = []
        for job in jobs:
            outbound = self.costs.network_ms(
                payload_bytes=job.payload_bytes,
                rng=self.rng,
                job_id=job.job_id,
                attempt=0,
                direction="request",
            )
            source.append(
                _Attempt(
                    job=job,
                    attempt=0,
                    network_ready_ms=job.submit_ms + outbound,
                )
            )
        source.sort(key=self._attempt_key)
        blocked: list[_Attempt] = []
        waiting: list[_Waiting] = []
        formations: dict[int, _Formation] = {}
        attempts: list[DraftAttemptRecord] = []
        batches: list[DraftBatchRecord] = []
        queue_peak = 0
        retries = 0
        batch_id = 0
        clock_ms = 0.0

        def admit_blocked(now_ms: float) -> bool:
            nonlocal queue_peak
            changed = False
            blocked.sort(key=self._attempt_key)
            while blocked and len(waiting) < self.config.queue_capacity:
                attempt = blocked.pop(0)
                waiting.append(_Waiting(attempt=attempt, admitted_ms=now_ms))
                queue_peak = max(queue_peak, len(waiting))
                changed = True
            return changed

        def receive_arrivals(now_ms: float) -> bool:
            changed = False
            while source and source[0].network_ready_ms <= now_ms:
                blocked.append(source.pop(0))
                changed = True
            return admit_blocked(now_ms) or changed

        def start_batch(
            worker_id: int,
            selected: list[_Waiting],
            now_ms: float,
        ) -> None:
            nonlocal batch_id, retries
            if not selected:
                raise AssertionError("cannot start an empty remote batch")
            jobs_in_batch = tuple(item.attempt.job for item in selected)
            kind = jobs_in_batch[0].kind
            if any(job.kind is not kind for job in jobs_in_batch):
                raise AssertionError("remote batch mixed incompatible job kinds")
            service_ms = self.costs.remote_service_ms(
                recovery=kind is DraftJobKind.RECOVERY,
                context_tokens=tuple(job.context_tokens for job in jobs_in_batch),
                branches=tuple(job.branches for job in jobs_in_batch),
            )
            end_ms = now_ms + service_ms
            batch_id += 1
            batches.append(
                DraftBatchRecord(
                    batch_id=batch_id,
                    worker_id=worker_id,
                    kind=kind,
                    job_ids=tuple(job.job_id for job in jobs_in_batch),
                    start_ms=now_ms,
                    end_ms=end_ms,
                )
            )
            worker_available[worker_id] = end_ms
            for waiting_attempt in selected:
                attempt = waiting_attempt.attempt
                job = attempt.job
                inbound = self.costs.network_ms(
                    payload_bytes=job.payload_bytes,
                    rng=self.rng,
                    job_id=job.job_id,
                    attempt=attempt.attempt,
                    direction="response",
                )
                response_ms = end_ms + inbound
                failed = self.rng.bernoulli(
                    self.config.failure_probability,
                    job.job_id,
                    attempt.attempt,
                    "remote-failure",
                )
                terminal = failed and attempt.attempt >= self.config.max_retries
                attempts.append(
                    DraftAttemptRecord(
                        job_id=job.job_id,
                        request_id=job.request_id,
                        kind=job.kind,
                        attempt=attempt.attempt,
                        worker_id=worker_id,
                        batch_id=batch_id,
                        network_ready_ms=attempt.network_ready_ms,
                        admitted_ms=waiting_attempt.admitted_ms,
                        service_start_ms=now_ms,
                        service_end_ms=end_ms,
                        response_ms=response_ms,
                        backpressure_delay_ms=max(
                            0.0,
                            waiting_attempt.admitted_ms - attempt.network_ready_ms,
                        ),
                        success=not failed,
                        terminal=terminal,
                    )
                )
                if failed and not terminal:
                    retries += 1
                    next_attempt = attempt.attempt + 1
                    outbound = self.costs.network_ms(
                        payload_bytes=job.payload_bytes,
                        rng=self.rng,
                        job_id=job.job_id,
                        attempt=next_attempt,
                        direction="request",
                    )
                    source.append(
                        _Attempt(
                            job=job,
                            attempt=next_attempt,
                            network_ready_ms=(
                                response_ms + self.config.retry_backoff_ms + outbound
                            ),
                        )
                    )
            source.sort(key=self._attempt_key)

        def fill_formations(now_ms: float) -> bool:
            changed = False
            for worker_id in sorted(
                formations,
                key=lambda index: (formations[index].deadline_ms, index),
            ):
                formation = formations[worker_id]
                while len(formation.selected) < self.config.max_batch_size:
                    compatible = next(
                        (
                            item
                            for item in sorted(waiting, key=self._waiting_key)
                            if item.attempt.job.kind is formation.kind
                        ),
                        None,
                    )
                    if compatible is None:
                        break
                    waiting.remove(compatible)
                    formation.selected.append(compatible)
                    changed = True
                    admit_blocked(now_ms)
                if (
                    len(formation.selected) >= self.config.max_batch_size
                    or formation.deadline_ms <= now_ms
                ):
                    formations.pop(worker_id)
                    start_batch(worker_id, formation.selected, now_ms)
                    changed = True
            return changed

        def dispatch_idle_workers(now_ms: float) -> bool:
            changed = False
            idle_workers = [
                worker_id
                for worker_id in range(self.config.workers)
                if worker_id not in formations and worker_available[worker_id] <= now_ms
            ]
            for worker_id in idle_workers:
                if not waiting:
                    break
                anchor = min(waiting, key=self._waiting_key)
                waiting.remove(anchor)
                admit_blocked(now_ms)
                if self.config.queue_policy == "priority":
                    start_batch(worker_id, [anchor], now_ms)
                else:
                    formations[worker_id] = _Formation(
                        kind=anchor.attempt.job.kind,
                        deadline_ms=now_ms + self.config.batch_window_ms,
                        selected=[anchor],
                    )
                    fill_formations(now_ms)
                changed = True
            return changed

        while source or blocked or waiting or formations:
            while True:
                before = (
                    len(source),
                    len(blocked),
                    len(waiting),
                    len(formations),
                    len(attempts),
                    len(batches),
                    tuple(worker_available),
                )
                receive_arrivals(clock_ms)
                fill_formations(clock_ms)
                dispatch_idle_workers(clock_ms)
                after = (
                    len(source),
                    len(blocked),
                    len(waiting),
                    len(formations),
                    len(attempts),
                    len(batches),
                    tuple(worker_available),
                )
                if after == before:
                    break

            future_events = [
                value
                for value in (
                    source[0].network_ready_ms if source else math.inf,
                    min(
                        (formation.deadline_ms for formation in formations.values()),
                        default=math.inf,
                    ),
                    min(
                        (
                            available_ms
                            for worker_id, available_ms in enumerate(worker_available)
                            if worker_id not in formations and available_ms > clock_ms
                        ),
                        default=math.inf,
                    ),
                )
                if math.isfinite(value) and value > clock_ms
            ]
            if not future_events:
                if source or blocked or waiting or formations:
                    raise AssertionError("remote scheduler reached a non-terminal fixed point")
                break
            clock_ms = min(future_events)

        successful = tuple(sorted(record.job_id for record in attempts if record.success))
        terminal_failed = tuple(sorted(record.job_id for record in attempts if record.terminal))
        trace = RemoteDraftTrace(
            attempts=tuple(attempts),
            batches=tuple(batches),
            successful_job_ids=successful,
            terminal_failed_job_ids=terminal_failed,
            worker_available_ms=tuple(worker_available),
            queue_peak=queue_peak,
            retries=retries,
            backpressured_attempts=sum(record.backpressure_delay_ms > 0.0 for record in attempts),
        )
        if len(successful) + len(terminal_failed) != len(jobs):
            raise AssertionError("remote service lost or duplicated a terminal job")
        if trace.queue_peak > self.config.queue_capacity:
            raise AssertionError("remote service exceeded queue capacity")
        return trace


@dataclass(frozen=True, slots=True)
class FidelityConfig:
    """Configuration for one executable one-round fidelity trace."""

    costs: ContextCostModel
    remote: RemoteDraftConfig
    cache_byte_budget: int
    cache_page_size_bytes: int
    kv_bytes_per_token: int
    continuation_tokens: int
    fanout: int
    target_batch_size: int

    def __post_init__(self) -> None:
        for field_name in (
            "cache_byte_budget",
            "cache_page_size_bytes",
            "kv_bytes_per_token",
            "continuation_tokens",
            "fanout",
            "target_batch_size",
        ):
            _positive_integer(getattr(self, field_name), field=field_name)

    @classmethod
    def reference(
        cls,
        profile: HardwareProfile,
        *,
        target_batch_size: int = 16,
    ) -> FidelityConfig:
        """Null transport/prefill configuration using existing latency methods."""

        return cls(
            costs=ContextCostModel.reference(profile),
            remote=RemoteDraftConfig.reference(max_batch_size=target_batch_size),
            cache_byte_budget=1 << 40,
            cache_page_size_bytes=1,
            kv_bytes_per_token=1,
            continuation_tokens=1,
            fanout=1,
            target_batch_size=target_batch_size,
        )


@dataclass(frozen=True, slots=True)
class PrefillRecord:
    request_id: str
    start_ms: float
    end_ms: float
    prompt_tokens: int


@dataclass(frozen=True, slots=True)
class TargetBatchRecord:
    request_ids: tuple[str, ...]
    start_ms: float
    end_ms: float
    verifier_slots: int


@dataclass(frozen=True, slots=True)
class FidelityRequestResult:
    request_id: str
    class_id: str
    accepted_tokens: int
    emitted_tokens: int
    terminal: bool
    realized_outcome: int | None
    cached_outcomes: tuple[int, ...]
    cache_hit: bool | None
    prefill_start_ms: float
    prefill_end_ms: float
    first_token_ms: float
    ttft_ms: float
    next_ready_ms: float | None
    output_tokens_remaining: int


@dataclass(frozen=True, slots=True)
class FidelityTrace:
    requests: tuple[FidelityRequestResult, ...]
    prefills: tuple[PrefillRecord, ...]
    target_batches: tuple[TargetBatchRecord, ...]
    precompute_trace: RemoteDraftTrace
    recovery_trace: RemoteDraftTrace | None
    cache_logical_bytes: int
    cache_allocated_pages: int
    cache_allocated_bytes: int
    cache_peak_allocated_pages: int
    cache_evictions: int
    cache_releases: int
    stale_precompute_jobs: int


def _schedule_prefill(
    resolved: tuple[ResolvedRequest, ...],
    costs: ContextCostModel,
) -> tuple[PrefillRecord, ...]:
    free_ms = 0.0
    records: list[PrefillRecord] = []
    for item in sorted(
        resolved,
        key=lambda entry: (entry.request.arrival_ms, entry.request.request_id),
    ):
        start_ms = max(free_ms, item.request.arrival_ms)
        end_ms = start_ms + costs.prefill_ms(item.request.prompt_tokens)
        records.append(
            PrefillRecord(
                request_id=item.request.request_id,
                start_ms=start_ms,
                end_ms=end_ms,
                prompt_tokens=item.request.prompt_tokens,
            )
        )
        free_ms = end_ms
    return tuple(records)


def _schedule_target(
    resolved: tuple[ResolvedRequest, ...],
    prefill_by_request: dict[str, PrefillRecord],
    config: FidelityConfig,
) -> tuple[TargetBatchRecord, ...]:
    pending = list(resolved)
    free_ms = 0.0
    records: list[TargetBatchRecord] = []
    while pending:
        eligible = [
            item
            for item in pending
            if prefill_by_request[item.request.request_id].end_ms <= free_ms
        ]
        if not eligible:
            free_ms = min(prefill_by_request[item.request.request_id].end_ms for item in pending)
            eligible = [
                item
                for item in pending
                if prefill_by_request[item.request.request_id].end_ms <= free_ms
            ]
        selected = sorted(
            eligible,
            key=lambda item: (
                prefill_by_request[item.request.request_id].end_ms,
                -item.request.priority,
                item.request.request_id,
            ),
        )[: config.target_batch_size]
        for item in selected:
            pending.remove(item)
        verifier_slots = sum(
            min(item.request.speculation_length, item.request.output_tokens) for item in selected
        )
        duration = config.costs.target_ms(
            context_tokens=tuple(item.request.prompt_tokens for item in selected),
            verifier_slots=verifier_slots,
        )
        end_ms = free_ms + duration
        records.append(
            TargetBatchRecord(
                request_ids=tuple(item.request.request_id for item in selected),
                start_ms=free_ms,
                end_ms=end_ms,
                verifier_slots=verifier_slots,
            )
        )
        free_ms = end_ms
    return tuple(records)


def simulate_fidelity_trace(
    requests: tuple[FidelityRequest, ...],
    classes: tuple[OutcomeClass, ...],
    config: FidelityConfig,
    *,
    seed: Seed,
) -> FidelityTrace:
    """Execute prefill, remote precompute, target lookup, and miss recovery.

    This trace intentionally models one speculative round. It is a fidelity
    transformer/service harness, not a competing online policy simulator.
    """

    resolved = resolve_request_classes(requests, classes, seed=seed)
    resolved_by_request = {item.request.request_id: item for item in resolved}
    prefills = _schedule_prefill(resolved, config.costs)
    prefill_by_request = {record.request_id: record for record in prefills}
    branch_logical_bytes = config.kv_bytes_per_token * config.continuation_tokens
    precompute_jobs = tuple(
        DraftJob(
            job_id=f"precompute/{item.request.request_id}/0",
            request_id=item.request.request_id,
            kind=DraftJobKind.PRECOMPUTE,
            submit_ms=prefill_by_request[item.request.request_id].end_ms,
            context_tokens=item.request.prompt_tokens,
            branches=min(
                config.fanout,
                len(item.outcome_class.outcome_probabilities),
            ),
            payload_bytes=(
                min(
                    config.fanout,
                    len(item.outcome_class.outcome_probabilities),
                )
                * branch_logical_bytes
            ),
            priority=item.request.priority,
        )
        for item in resolved
    )
    precompute_trace = RemoteDraftService(
        config.remote,
        config.costs,
        seed=seed,
    ).run(precompute_jobs)
    targets = _schedule_target(resolved, prefill_by_request, config)
    target_by_request = {request_id: batch for batch in targets for request_id in batch.request_ids}
    completion_by_job = precompute_trace.completion_by_job()
    cache = OutcomeTreeCache(
        byte_budget=config.cache_byte_budget,
        page_size_bytes=config.cache_page_size_bytes,
    )
    events: list[tuple[float, int, str, str]] = []
    for job in precompute_jobs:
        completion = completion_by_job[job.job_id]
        if completion.success:
            events.append(
                (
                    completion.response_ms,
                    0,
                    "precompute",
                    job.request_id,
                )
            )
    for item in resolved:
        events.append(
            (
                target_by_request[item.request.request_id].end_ms,
                1,
                "lookup",
                item.request.request_id,
            )
        )
    accepted_by_request = {
        item.request.request_id: sample_accepted_tokens(
            item,
            round_id=0,
            seed=seed,
        )
        for item in resolved
    }
    emitted_by_request = {
        item.request.request_id: min(
            item.request.output_tokens,
            accepted_by_request[item.request.request_id] + 1,
        )
        for item in resolved
    }
    terminal_by_request = {
        item.request.request_id: (
            emitted_by_request[item.request.request_id] == item.request.output_tokens
        )
        for item in resolved
    }
    resident_outcomes_by_request: dict[str, tuple[int, ...]] = {}
    outcome_by_request = {
        item.request.request_id: sample_realized_outcome(
            item,
            round_id=0,
            seed=seed,
        )
        for item in resolved
        if not terminal_by_request[item.request.request_id]
    }
    looked_up: set[str] = set()
    hit_by_request: dict[str, bool | None] = {}
    evictions = 0
    releases = 0
    stale_precompute = 0
    for _, _, event_kind, request_id in sorted(events):
        item = resolved_by_request[request_id]
        planned_outcomes = item.outcome_class.popular_outcomes(config.fanout)
        if event_kind == "precompute":
            if request_id in looked_up:
                stale_precompute += 1
                continue
            # Insert least-popular first so the most popular branches are most
            # recent if this tree itself exceeds the remaining budget.
            for outcome in reversed(planned_outcomes):
                mutation = cache.insert(
                    (request_id, 0, outcome),
                    logical_bytes=branch_logical_bytes,
                )
                evictions += len(mutation.evicted_keys)
        else:
            looked_up.add(request_id)
            if terminal_by_request[request_id]:
                resident_outcomes_by_request[request_id] = ()
                hit_by_request[request_id] = None
                releases += sum(
                    cache.discard((request_id, 0, outcome)) for outcome in planned_outcomes
                )
            else:
                resident_outcomes_by_request[request_id] = tuple(
                    outcome
                    for outcome in planned_outcomes
                    if cache.contains((request_id, 0, outcome))
                )
                hit_by_request[request_id] = cache.lookup(
                    (request_id, 0, outcome_by_request[request_id])
                )
        cache.audit()

    misses = tuple(item for item in resolved if hit_by_request[item.request.request_id] is False)
    recovery_trace: RemoteDraftTrace | None = None
    recovery_by_request: dict[str, DraftAttemptRecord] = {}
    if misses:
        recovery_jobs = tuple(
            DraftJob(
                job_id=f"recovery/{item.request.request_id}/0",
                request_id=item.request.request_id,
                kind=DraftJobKind.RECOVERY,
                submit_ms=target_by_request[item.request.request_id].end_ms,
                context_tokens=item.request.prompt_tokens,
                branches=1,
                payload_bytes=branch_logical_bytes,
                priority=item.request.priority,
            )
            for item in misses
        )
        recovery_trace = RemoteDraftService(
            config.remote,
            config.costs,
            seed=seed,
        ).run(
            recovery_jobs,
            initial_worker_available_ms=precompute_trace.worker_available_ms,
        )
        recovery_by_request = {
            record.request_id: record for record in recovery_trace.completion_by_job().values()
        }

    results: list[FidelityRequestResult] = []
    for item in sorted(resolved, key=lambda entry: entry.request.request_id):
        request = item.request
        target = target_by_request[request.request_id]
        accepted = accepted_by_request[request.request_id]
        emitted = emitted_by_request[request.request_id]
        terminal = terminal_by_request[request.request_id]
        hit = hit_by_request[request.request_id]
        if terminal:
            next_ready_ms: float | None = None
        elif hit:
            next_ready_ms = target.end_ms
        else:
            recovery = recovery_by_request[request.request_id]
            next_ready_ms = recovery.response_ms if recovery.success else None
        prefill = prefill_by_request[request.request_id]
        results.append(
            FidelityRequestResult(
                request_id=request.request_id,
                class_id=item.outcome_class.class_id,
                accepted_tokens=min(accepted, emitted),
                emitted_tokens=emitted,
                terminal=terminal,
                realized_outcome=outcome_by_request.get(request.request_id),
                cached_outcomes=resident_outcomes_by_request[request.request_id],
                cache_hit=hit,
                prefill_start_ms=prefill.start_ms,
                prefill_end_ms=prefill.end_ms,
                first_token_ms=target.end_ms,
                ttft_ms=target.end_ms - request.arrival_ms,
                next_ready_ms=next_ready_ms,
                output_tokens_remaining=request.output_tokens - emitted,
            )
        )
    return FidelityTrace(
        requests=tuple(results),
        prefills=prefills,
        target_batches=targets,
        precompute_trace=precompute_trace,
        recovery_trace=recovery_trace,
        cache_logical_bytes=cache.logical_bytes,
        cache_allocated_pages=cache.allocated_pages,
        cache_allocated_bytes=cache.allocated_bytes,
        cache_peak_allocated_pages=cache.peak_allocated_pages,
        cache_evictions=evictions,
        cache_releases=releases,
        stale_precompute_jobs=stale_precompute,
    )
