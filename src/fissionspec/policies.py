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
    """

    now_ms: float
    ready_count: int
    capacity: int
    oldest_ready_ms: float
    earliest_deadline_ms: float
    slots_per_row: int
    profile: HardwareProfile
    next_ready_time_ms: float | None = None
    next_ready_count: int = 0
    earliest_future_deadline_ms: float | None = None
    next_slots_per_row: int | None = None

    def __post_init__(self) -> None:
        if self.ready_count <= 0:
            raise ValueError("ready_count must be positive")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.slots_per_row <= 0:
            raise ValueError("slots_per_row must be positive")
        if self.next_ready_count < 0:
            raise ValueError("next_ready_count must be non-negative")
        if self.next_slots_per_row is not None and self.next_slots_per_row <= 0:
            raise ValueError("next_slots_per_row must be positive when supplied")


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
        if not math.isfinite(self.coalesce_ms) or self.coalesce_ms < 0.0:
            raise ValueError("coalesce_ms must be finite and non-negative")

    def dispatch_at(self, context: DispatchContext) -> float:
        if context.ready_count >= context.capacity:
            return context.now_ms
        normal_due = max(context.now_ms, context.oldest_ready_ms + self.coalesce_ms)
        # Never knowingly turn a batching window into a deadline violation.
        latest_safe = context.earliest_deadline_ms - context.profile.target_latency_ms(
            min(context.ready_count, context.capacity),
            min(context.ready_count, context.capacity) * context.slots_per_row,
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
        if not math.isfinite(self.max_wait_ms) or self.max_wait_ms < 0.0:
            raise ValueError("max_wait_ms must be finite and non-negative")

    def dispatch_at(self, context: DispatchContext) -> float:
        if context.ready_count >= context.capacity:
            return context.now_ms
        next_time = context.next_ready_time_ms
        if next_time is None or context.next_ready_count == 0:
            return context.now_ms
        delay = max(0.0, next_time - context.now_ms)
        hard_wait_deadline = context.oldest_ready_ms + self.max_wait_ms
        if (
            delay <= 0.0
            or context.now_ms >= hard_wait_deadline
            or next_time > hard_wait_deadline
        ):
            return context.now_ms

        now_rows = context.ready_count
        future_rows = context.next_ready_count
        future_slots_per_row = (
            context.next_slots_per_row
            if context.next_slots_per_row is not None
            else context.slots_per_row
        )
        future_deadline = (
            context.earliest_future_deadline_ms
            if context.earliest_future_deadline_ms is not None
            else math.inf
        )
        first_latency = context.profile.target_latency_ms(
            now_rows, now_rows * context.slots_per_row
        )

        # Launch-now aggregate flow cost.  Future chunks arrive at ``delta``
        # but cannot use the target until its current launch has completed.
        target_available = max(next_time, context.now_ms + first_latency)
        launch_now_cost = now_rows * first_latency
        launch_now_feasible = (
            context.now_ms + first_latency <= context.earliest_deadline_ms
        )
        remaining_future = future_rows
        while remaining_future > 0:
            chunk = min(remaining_future, context.capacity)
            target_available += context.profile.target_latency_ms(
                chunk, chunk * future_slots_per_row
            )
            launch_now_cost += chunk * (target_available - next_time)
            launch_now_feasible = (
                launch_now_feasible and target_available <= future_deadline
            )
            remaining_future -= chunk

        # Wait aggregate flow cost.  Ready rows retain priority in the first
        # coalesced chunk, so only future rows can overflow behind them.
        first_wait_rows = min(now_rows + future_rows, context.capacity)
        future_in_first = max(0, first_wait_rows - now_rows)
        first_wait_slots = (
            now_rows * context.slots_per_row
            + future_in_first * future_slots_per_row
        )
        first_wait_latency = context.profile.target_latency_ms(
            first_wait_rows, first_wait_slots
        )
        wait_completion = next_time + first_wait_latency
        wait_cost = now_rows * (wait_completion - context.now_ms)
        wait_cost += future_in_first * (wait_completion - next_time)
        wait_feasible = wait_completion <= context.earliest_deadline_ms
        if future_in_first:
            wait_feasible = wait_feasible and wait_completion <= future_deadline
        remaining_future = future_rows - future_in_first
        while remaining_future > 0:
            chunk = min(remaining_future, context.capacity)
            wait_completion += context.profile.target_latency_ms(
                chunk, chunk * future_slots_per_row
            )
            wait_cost += chunk * (wait_completion - next_time)
            wait_feasible = wait_feasible and wait_completion <= future_deadline
            remaining_future -= chunk

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
