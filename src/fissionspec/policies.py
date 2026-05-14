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

    def __post_init__(self) -> None:
        if self.ready_count <= 0:
            raise ValueError("ready_count must be positive")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.slots_per_row <= 0:
            raise ValueError("slots_per_row must be positive")
        if self.next_ready_count < 0:
            raise ValueError("next_ready_count must be non-negative")


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
    """Cohort barrier: one miss sends every surviving row through recovery."""

    name: str = "saguaro-barrier"
    barrier_on_miss: bool = True
    pad_recovering_misses: bool = False

    def dispatch_at(self, context: DispatchContext) -> float:
        return context.now_ms


@dataclass(frozen=True, slots=True)
class SPECTREPaddedPolicy:
    """Mixed semantics: recovering misses occupy one-token padded target rows."""

    name: str = "spectre-padded"
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

    @staticmethod
    def _latency(context: DispatchContext, rows: int) -> float:
        return context.profile.target_latency_ms(rows, rows * context.slots_per_row)

    def dispatch_at(self, context: DispatchContext) -> float:
        if context.ready_count >= context.capacity:
            return context.now_ms
        next_time = context.next_ready_time_ms
        if next_time is None or context.next_ready_count == 0:
            return context.now_ms
        delay = max(0.0, next_time - context.now_ms)
        if delay <= 0.0 or delay > self.max_wait_ms:
            return context.now_ms

        now_rows = context.ready_count
        future_rows = context.next_ready_count
        first_latency = self._latency(context, now_rows)

        # Launch-now aggregate flow cost.  Future chunks arrive at ``delta``
        # but cannot use the target until its current launch has completed.
        target_available = max(delay, first_latency)
        launch_now_cost = now_rows * first_latency
        remaining_future = future_rows
        while remaining_future > 0:
            chunk = min(remaining_future, context.capacity)
            target_available += self._latency(context, chunk)
            launch_now_cost += chunk * (target_available - delay)
            remaining_future -= chunk

        # Wait aggregate flow cost.  Ready rows retain priority in the first
        # coalesced chunk, so only future rows can overflow behind them.
        first_wait_rows = min(now_rows + future_rows, context.capacity)
        first_wait_latency = self._latency(context, first_wait_rows)
        wait_completion = delay + first_wait_latency
        wait_cost = now_rows * wait_completion
        future_in_first = max(0, first_wait_rows - now_rows)
        wait_cost += future_in_first * first_wait_latency
        remaining_future = future_rows - future_in_first
        while remaining_future > 0:
            chunk = min(remaining_future, context.capacity)
            wait_completion += self._latency(context, chunk)
            wait_cost += chunk * (wait_completion - delay)
            remaining_future -= chunk

        predicted_completion = next_time + first_wait_latency
        misses_deadline = predicted_completion > context.earliest_deadline_ms
        return (
            context.now_ms
            if misses_deadline or wait_cost >= launch_now_cost
            else next_time
        )


Horizon2Policy = FissionSpecPolicy


def policy_from_name(
    name: str, *, coalesce_ms: float = 1.0, max_wait_ms: float = 2.0
) -> SchedulingPolicy:
    """Parse the stable CLI names for built-in policies."""

    normalized = name.strip().lower().replace("_", "-")
    if normalized in {"saguaro", "saguaro-barrier", "barrier"}:
        return SaguaroBarrierPolicy()
    if normalized in {"spectre", "spectre-padded", "padded"}:
        return SPECTREPaddedPolicy()
    if normalized in {"immediate", "immediate-fission", "fission"}:
        return ImmediateFissionPolicy()
    if normalized in {"fixed", "fixed-coalesce", "coalesce"}:
        return FixedCoalescePolicy(coalesce_ms=coalesce_ms)
    if normalized in {"fissionspec", "horizon-2", "fissionspec-horizon-2"}:
        return FissionSpecPolicy(max_wait_ms=max_wait_ms)
    raise ValueError(f"unknown policy: {name!r}")
