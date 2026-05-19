"""Scheduling policies implemented by the FissionSpec reference simulator."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .profiles import HardwareProfile


@dataclass(frozen=True, slots=True)
class DispatchContext:
    """Information available to a policy at an idle target engine.

    The horizon intentionally exposes only the next readiness event.  Looking
    through two target launches is enough to price the launch-now versus
    coalesce decision while avoiding oracle knowledge of future outcomes.
    Slot and deadline tuples are paired in exact rolling-EDF admission order
    within each cohort.  That is the minimum metadata needed to reproduce the
    global admission merge at a prospective wake without exposing request
    identities.
    """

    now_ms: float
    ready_count: int
    capacity: int
    oldest_ready_ms: float
    earliest_deadline_ms: float
    row_slots: tuple[int, ...]
    row_deadlines_ms: tuple[float, ...]
    profile: HardwareProfile
    next_ready_time_ms: float | None = None
    next_ready_count: int = 0
    earliest_future_deadline_ms: float | None = None
    future_row_slots: tuple[int, ...] = ()
    future_row_deadlines_ms: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.ready_count, bool)
            or not isinstance(self.ready_count, int)
            or self.ready_count <= 0
        ):
            raise ValueError("ready_count must be positive")
        if (
            isinstance(self.capacity, bool)
            or not isinstance(self.capacity, int)
            or self.capacity <= 0
        ):
            raise ValueError("capacity must be positive")
        if (
            isinstance(self.next_ready_count, bool)
            or not isinstance(self.next_ready_count, int)
            or self.next_ready_count < 0
        ):
            raise ValueError("next_ready_count must be non-negative")
        for name, value in (
            ("now_ms", self.now_ms),
            ("oldest_ready_ms", self.oldest_ready_ms),
            ("earliest_deadline_ms", self.earliest_deadline_ms),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if self.oldest_ready_ms > self.now_ms:
            raise ValueError("oldest_ready_ms must not be in the future")
        if self.next_ready_count and (
            isinstance(self.next_ready_time_ms, bool)
            or not isinstance(self.next_ready_time_ms, (int, float))
            or not math.isfinite(self.next_ready_time_ms)
            or self.next_ready_time_ms < self.now_ms
        ):
            raise ValueError("next_ready_time_ms must be a finite future readiness time")
        selected_rows = min(self.ready_count, self.capacity)
        if len(self.row_slots) != selected_rows or any(
            isinstance(slots, bool) or not isinstance(slots, int) or slots <= 0
            for slots in self.row_slots
        ):
            raise ValueError("row_slots must contain one positive width per selected row")
        if len(self.row_deadlines_ms) != selected_rows or any(
            isinstance(deadline, bool)
            or not isinstance(deadline, (int, float))
            or not math.isfinite(deadline)
            for deadline in self.row_deadlines_ms
        ):
            raise ValueError("row_deadlines_ms must contain one finite deadline per selected row")
        if any(
            left > right
            for left, right in zip(
                self.row_deadlines_ms,
                self.row_deadlines_ms[1:],
                strict=False,
            )
        ):
            raise ValueError("row_deadlines_ms must be in target-admission order")
        if self.earliest_deadline_ms != self.row_deadlines_ms[0]:
            raise ValueError("earliest_deadline_ms must match the first ordered row deadline")
        if len(self.future_row_slots) != self.next_ready_count or any(
            isinstance(slots, bool) or not isinstance(slots, int) or slots <= 0
            for slots in self.future_row_slots
        ):
            raise ValueError("future_row_slots must contain one positive width per future row")
        if len(self.future_row_deadlines_ms) != self.next_ready_count or any(
            isinstance(deadline, bool)
            or not isinstance(deadline, (int, float))
            or not math.isfinite(deadline)
            for deadline in self.future_row_deadlines_ms
        ):
            raise ValueError(
                "future_row_deadlines_ms must contain one finite deadline per future row"
            )
        if any(
            left > right
            for left, right in zip(
                self.future_row_deadlines_ms,
                self.future_row_deadlines_ms[1:],
                strict=False,
            )
        ):
            raise ValueError("future_row_deadlines_ms must be in target-admission order")
        if self.next_ready_count and (
            self.earliest_future_deadline_ms is None
            or self.earliest_future_deadline_ms != self.future_row_deadlines_ms[0]
        ):
            raise ValueError(
                "earliest_future_deadline_ms must match the first ordered future deadline"
            )


@dataclass(frozen=True, slots=True)
class _ForecastRow:
    """One row in the exact target-admission forecast."""

    deadline_ms: float
    verifier_slots: int
    is_future: bool
    cohort_ordinal: int


@runtime_checkable
class SchedulingPolicy(Protocol):
    """Structural interface consumed by :class:`fissionspec.simulator.Simulator`."""

    @property
    def name(self) -> str: ...

    @property
    def barrier_on_miss(self) -> bool: ...

    @property
    def pad_recovering_misses(self) -> bool: ...

    def dispatch_at(self, context: DispatchContext) -> float:
        """Return the earliest absolute time at which target work should launch."""


@dataclass(frozen=True, slots=True)
class SaguaroBarrierPolicy:
    """Miss-only fallback that holds all surviving rows behind a cohort barrier."""

    name: str = "saguaro-barrier"
    barrier_on_miss: bool = True
    pad_recovering_misses: bool = False

    def dispatch_at(self, context: DispatchContext) -> float:
        return context.now_ms


@dataclass(frozen=True, slots=True)
class SPECTREPaddedPolicy:
    """SPECTRE parallel-mode semantics with one-token padded recovery rows.

    This is intentionally not SPECTRE's hybrid selector.  The simulator models
    one padded target step per recovery chain, then fences a stale recovery and
    completes its version repair without another padded step.
    """

    name: str = "spectre-parallel-padded"
    barrier_on_miss: bool = False
    pad_recovering_misses: bool = True

    def dispatch_at(self, context: DispatchContext) -> float:
        return context.now_ms


# Conventional spelling retained in addition to the paper-style acronym.
SpectrePaddedPolicy = SPECTREPaddedPolicy


@dataclass(frozen=True, slots=True)
class ImmediateFissionPolicy:
    """Outcome decoupling with work-conserving target dispatch."""

    name: str = "immediate-fission"
    barrier_on_miss: bool = False
    pad_recovering_misses: bool = False

    def dispatch_at(self, context: DispatchContext) -> float:
        return context.now_ms


@dataclass(frozen=True, slots=True)
class FixedCoalescePolicy:
    """Fission followed by a fixed batching window from the oldest ready row."""

    coalesce_ms: float = 1.0
    name: str = "fixed-coalesce"
    barrier_on_miss: bool = False
    pad_recovering_misses: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.coalesce_ms, bool)
            or not isinstance(self.coalesce_ms, (int, float))
            or not math.isfinite(self.coalesce_ms)
            or self.coalesce_ms < 0.0
        ):
            raise ValueError("coalesce_ms must be finite and non-negative")

    def dispatch_at(self, context: DispatchContext) -> float:
        if context.ready_count >= context.capacity:
            return context.now_ms
        normal_due = max(context.now_ms, context.oldest_ready_ms + self.coalesce_ms)
        # Never knowingly turn a batching window into a deadline violation.
        latest_safe = context.earliest_deadline_ms - context.profile.target_latency_ms(
            min(context.ready_count, context.capacity),
            sum(context.row_slots),
        )
        return max(context.now_ms, min(normal_due, latest_safe))


@dataclass(frozen=True, slots=True)
class FissionSpecPolicy:
    """Deadline-aware horizon-2 outcome-decoupled controller.

    At each idle point the controller minimizes aggregate flow time across the
    current and next readiness sets.  For sets of size ``n`` and ``m`` that fit
    together, its exact comparison is::

        C_now  = n L(n) + m (max(L(n), delta) + L(m) - delta)
        C_wait = n (delta + L(n + m)) + m L(n + m)

    Capacity overflow is priced as successive target chunks.  This is a
    horizon-2 policy because it sees only the next readiness set, never future
    outcomes or later arrivals.
    """

    max_wait_ms: float = 2.0
    name: str = "fissionspec-horizon-2"
    barrier_on_miss: bool = False
    pad_recovering_misses: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_wait_ms, bool)
            or not isinstance(self.max_wait_ms, (int, float))
            or not math.isfinite(self.max_wait_ms)
            or self.max_wait_ms < 0.0
        ):
            raise ValueError("max_wait_ms must be finite and non-negative")

    def dispatch_at(self, context: DispatchContext) -> float:
        if context.ready_count >= context.capacity:
            return context.now_ms
        next_time = context.next_ready_time_ms
        if next_time is None or context.next_ready_count == 0:
            return context.now_ms
        delay = max(0.0, next_time - context.now_ms)
        hard_wait_deadline = context.oldest_ready_ms + self.max_wait_ms
        if delay <= 0.0 or context.now_ms >= hard_wait_deadline or next_time > hard_wait_deadline:
            return context.now_ms

        now_rows = context.ready_count
        future_rows = context.next_ready_count
        current = tuple(
            _ForecastRow(
                deadline_ms=deadline_ms,
                verifier_slots=slots,
                is_future=False,
                cohort_ordinal=index,
            )
            for index, (deadline_ms, slots) in enumerate(
                zip(context.row_deadlines_ms, context.row_slots, strict=True)
            )
        )
        future = tuple(
            _ForecastRow(
                deadline_ms=deadline_ms,
                verifier_slots=slots,
                is_future=True,
                cohort_ordinal=index,
            )
            for index, (deadline_ms, slots) in enumerate(
                zip(
                    context.future_row_deadlines_ms,
                    context.future_row_slots,
                    strict=True,
                )
            )
        )
        first_latency = context.profile.target_latency_ms(now_rows, sum(context.row_slots))

        # Launch-now aggregate flow cost.  Future chunks arrive at ``delta``
        # but cannot use the target until its current launch has completed.
        target_available = max(next_time, context.now_ms + first_latency)
        launch_now_cost = now_rows * first_latency
        current_completion = context.now_ms + first_latency
        launch_now_feasible = all(current_completion <= row.deadline_ms for row in current)
        for offset in range(0, future_rows, context.capacity):
            chunk = future[offset : offset + context.capacity]
            target_available += context.profile.target_latency_ms(
                len(chunk),
                sum(row.verifier_slots for row in chunk),
            )
            launch_now_cost += len(chunk) * (target_available - next_time)
            launch_now_feasible = launch_now_feasible and all(
                target_available <= row.deadline_ms for row in chunk
            )

        # At the wake, the simulator globally re-runs rolling EDF admission.
        # Each cohort is already in exact admission order. Current rows win
        # cross-cohort deadline ties because their ready_since timestamp
        # precedes the next readiness event.
        wait_order = tuple(
            sorted(
                current + future,
                key=lambda row: (
                    row.deadline_ms,
                    row.is_future,
                    row.cohort_ordinal,
                ),
            )
        )
        wait_completion = next_time
        wait_cost = 0.0
        wait_feasible = True
        for offset in range(0, len(wait_order), context.capacity):
            chunk = wait_order[offset : offset + context.capacity]
            wait_completion += context.profile.target_latency_ms(
                len(chunk),
                sum(row.verifier_slots for row in chunk),
            )
            for row in chunk:
                ready_at_ms = next_time if row.is_future else context.now_ms
                wait_cost += wait_completion - ready_at_ms
                wait_feasible = wait_feasible and wait_completion <= row.deadline_ms

        if not wait_feasible:
            return context.now_ms
        if not launch_now_feasible:
            return next_time
        return next_time if wait_cost < launch_now_cost else context.now_ms


Horizon2Policy = FissionSpecPolicy


def policy_from_name(
    name: str, *, coalesce_ms: float = 1.0, max_wait_ms: float = 2.0
) -> SchedulingPolicy:
    """Parse the stable CLI names for built-in policies."""

    normalized = name.strip().lower().replace("_", "-")
    if normalized in {"saguaro", "saguaro-barrier", "barrier"}:
        return SaguaroBarrierPolicy()
    if normalized in {
        "spectre",
        "spectre-padded",
        "spectre-parallel-padded",
        "padded",
    }:
        return SPECTREPaddedPolicy()
    if normalized in {"immediate", "immediate-fission", "fission"}:
        return ImmediateFissionPolicy()
    if normalized in {"fixed", "fixed-coalesce", "coalesce"}:
        return FixedCoalescePolicy(coalesce_ms=coalesce_ms)
    if normalized in {"fissionspec", "horizon-2", "fissionspec-horizon-2"}:
        return FissionSpecPolicy(max_wait_ms=max_wait_ms)
    raise ValueError(f"unknown policy: {name!r}")
