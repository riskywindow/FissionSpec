"""Core trace types shared by the simulator and metric layer."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .profiles import HardwareProfile
    from .workload import Workload


class Outcome(StrEnum):
    """Next-continuation state after one target verification."""

    HIT = "hit"
    MISS = "miss"
    TERMINAL = "terminal"


class RequestPhase(StrEnum):
    """Lifecycle phase of a simulated request."""

    NOT_ARRIVED = "not_arrived"
    WAIT_TARGET = "wait_target"
    IN_TARGET = "in_target"
    WAIT_DRAFT = "wait_draft"
    COMPLETE = "complete"


@dataclass(slots=True)
class RequestState:
    """Mutable state owned exclusively by a :class:`Simulator` instance."""

    request_id: str
    arrival_ms: float
    output_tokens: int
    speculation_length: int
    tbt_slo_ms: float
    absolute_deadline_ms: float
    phase: RequestPhase = RequestPhase.NOT_ARRIVED
    generated_tokens: int = 0
    logical_version: int = 0
    round_id: int = 0
    ready_since_ms: float | None = None
    completion_ms: float | None = None
    token_times_ms: list[float] = field(default_factory=list)
    hits: int = 0
    misses: int = 0
    accepted_draft_tokens: int = 0
    verifier_emitted_tokens: int = 0
    direct_hit_delay_ms: float = 0.0
    recovery_epoch: int = 0
    spectre_padding_eligible: bool = False
    waiting_precompute_round: int | None = None

    @property
    def remaining_tokens(self) -> int:
        return self.output_tokens - self.generated_tokens

    @property
    def next_token_deadline_ms(self) -> float:
        """Return the rolling TBT deadline capped by the final deadline.

        Before the first emitted token only the explicit/derived final bound
        applies because this steady-state model reports TBT but excludes TTFT.
        """

        if not self.token_times_ms:
            return self.absolute_deadline_ms
        return min(
            self.absolute_deadline_ms,
            self.token_times_ms[-1] + self.tbt_slo_ms,
        )

    def emit(self, count: int, at_ms: float) -> int:
        """Emit at most the remaining token count and return what was emitted."""

        if count <= 0 or not math.isfinite(at_ms):
            raise ValueError("emission count and timestamp must be valid")
        emitted = min(count, self.remaining_tokens)
        self.token_times_ms.extend(at_ms for _ in range(emitted))
        self.generated_tokens += emitted
        self.logical_version += emitted
        if self.generated_tokens == self.output_tokens:
            self.phase = RequestPhase.COMPLETE
            self.completion_ms = at_ms
        return emitted


@dataclass(frozen=True, slots=True)
class RequestResult:
    """Immutable per-request result returned to callers."""

    request_id: str
    arrival_ms: float
    completion_ms: float
    output_tokens: int
    token_times_ms: tuple[float, ...]
    hits: int
    misses: int
    accepted_draft_tokens: int
    verifier_emitted_tokens: int
    tbt_slo_ms: float
    direct_hit_delay_ms: float

    @property
    def latency_ms(self) -> float:
        return self.completion_ms - self.arrival_ms

    @property
    def inter_token_times_ms(self) -> tuple[float, ...]:
        return tuple(
            right - left
            for left, right in zip(self.token_times_ms, self.token_times_ms[1:], strict=False)
        )


@dataclass(frozen=True, slots=True)
class TargetLaunchRecord:
    """One target-model verifier launch."""

    launch_id: int
    start_ms: float
    end_ms: float
    request_ids: tuple[str, ...]
    padded_request_ids: tuple[str, ...]
    outcomes: tuple[tuple[str, Outcome], ...]
    accepted_tokens: tuple[tuple[str, int], ...]
    productive_tokens: tuple[tuple[str, int], ...]
    verifier_slots: int
    padded_verifier_slots: int

    @property
    def real_batch_size(self) -> int:
        return len(self.request_ids)

    @property
    def effective_batch_size(self) -> int:
        return len(self.request_ids) + len(self.padded_request_ids)

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


@dataclass(frozen=True, slots=True)
class DraftLaunchRecord:
    """One draft-model launch on its independent timeline."""

    launch_id: int
    start_ms: float
    end_ms: float
    request_ids: tuple[str, ...]
    recovery: bool
    barrier: bool = False
    barrier_victim_ids: tuple[str, ...] = ()
    precompute: bool = False

    @property
    def batch_size(self) -> int:
        return len(self.request_ids)

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Complete deterministic trace for one policy/workload pair."""

    policy_name: str
    hardware_name: str
    workload_name: str
    rng_provenance: str
    profile: HardwareProfile
    workload: Workload
    requests: tuple[RequestResult, ...]
    target_launches: tuple[TargetLaunchRecord, ...]
    draft_launches: tuple[DraftLaunchRecord, ...]
    started_ms: float
    finished_ms: float

    @property
    def makespan_ms(self) -> float:
        return self.finished_ms - self.started_ms

    @property
    def completed_requests(self) -> int:
        return len(self.requests)

    @property
    def total_output_tokens(self) -> int:
        return sum(request.output_tokens for request in self.requests)

    @property
    def padded_verifier_slots(self) -> int:
        return sum(launch.padded_verifier_slots for launch in self.target_launches)
