"""Hardware latency models used by the dependency-free simulator."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class LatencyCurve:
    """Monotone piecewise-linear latency curve indexed by batch rows.

    Interpolation makes small research sweeps possible without a numerical
    dependency.  Values beyond the measured range are linearly extrapolated
    using the last segment; a single-point curve scales linearly.
    """

    points: tuple[tuple[int, float], ...]

    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError("a latency curve needs at least one point")
        previous_batch = 0
        previous_latency = -1.0
        for batch, latency in self.points:
            if (
                isinstance(batch, bool)
                or not isinstance(batch, int)
                or batch <= previous_batch
            ):
                raise ValueError("curve batch sizes must be strictly increasing")
            if not math.isfinite(latency) or latency <= 0.0:
                raise ValueError("curve latencies must be finite and positive")
            if latency < previous_latency:
                raise ValueError("curve latencies must be monotone")
            previous_batch = batch
            previous_latency = latency

    @classmethod
    def from_pairs(cls, points: Iterable[tuple[int, float]]) -> LatencyCurve:
        return cls(tuple(points))

    @classmethod
    def linear(cls, intercept_ms: float, per_row_ms: float) -> LatencyCurve:
        """Construct a linear curve from two representative samples."""

        if intercept_ms < 0.0 or per_row_ms <= 0.0:
            raise ValueError("intercept must be non-negative and slope positive")
        return cls(((1, intercept_ms + per_row_ms), (2, intercept_ms + 2 * per_row_ms)))

    def latency_ms(self, batch_rows: int) -> float:
        """Interpolate or extrapolate latency at ``batch_rows``."""

        if (
            isinstance(batch_rows, bool)
            or not isinstance(batch_rows, int)
            or batch_rows <= 0
        ):
            raise ValueError("batch_rows must be a positive integer")
        first_x, first_y = self.points[0]
        if batch_rows <= first_x:
            return first_y * batch_rows / first_x

        for (left_x, left_y), (right_x, right_y) in zip(
            self.points, self.points[1:], strict=False
        ):
            if batch_rows <= right_x:
                fraction = (batch_rows - left_x) / (right_x - left_x)
                return left_y + fraction * (right_y - left_y)

        if len(self.points) == 1:
            return first_y * batch_rows / first_x
        left_x, left_y = self.points[-2]
        right_x, right_y = self.points[-1]
        slope = (right_y - left_y) / (right_x - left_x)
        return right_y + (batch_rows - right_x) * slope

    __call__ = latency_ms


def _default_target_curve() -> LatencyCurve:
    return LatencyCurve(((1, 2.1), (4, 2.8), (8, 3.8), (16, 5.9), (32, 10.5)))


def _default_draft_curve() -> LatencyCurve:
    return LatencyCurve(((1, 0.55), (4, 0.78), (8, 1.05), (16, 1.65), (32, 2.85)))


def _default_recovery_curve() -> LatencyCurve:
    return LatencyCurve(((1, 1.25), (4, 1.7), (8, 2.3), (16, 3.5), (32, 5.9)))


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    """Latency curves for physically separate target and draft engines.

    Curves capture row-dependent kernel efficiency.  ``verifier_slot_ms``
    captures the smaller token-axis cost, including slots wasted by padded
    mixed batches.  Draft recovery has a separate curve because cache repair
    is normally more expensive than preparing the next speculative block.
    """

    target_curve: LatencyCurve = field(default_factory=_default_target_curve)
    draft_curve: LatencyCurve = field(default_factory=_default_draft_curve)
    recovery_curve: LatencyCurve = field(default_factory=_default_recovery_curve)
    verifier_slot_ms: float = 0.018
    name: str = "reference"

    def __post_init__(self) -> None:
        if not math.isfinite(self.verifier_slot_ms) or self.verifier_slot_ms < 0.0:
            raise ValueError("verifier_slot_ms must be finite and non-negative")

    @classmethod
    def linear(
        cls,
        *,
        target_overhead_ms: float = 1.0,
        target_per_row_ms: float = 0.2,
        draft_overhead_ms: float = 0.2,
        draft_per_row_ms: float = 0.05,
        recovery_overhead_ms: float = 0.5,
        recovery_per_row_ms: float = 0.1,
        verifier_slot_ms: float = 0.0,
        name: str = "linear",
    ) -> HardwareProfile:
        """Convenience constructor for analytically transparent tests."""

        return cls(
            target_curve=LatencyCurve.linear(target_overhead_ms, target_per_row_ms),
            draft_curve=LatencyCurve.linear(draft_overhead_ms, draft_per_row_ms),
            recovery_curve=LatencyCurve.linear(
                recovery_overhead_ms, recovery_per_row_ms
            ),
            verifier_slot_ms=verifier_slot_ms,
            name=name,
        )

    def target_latency_ms(self, batch_rows: int, verifier_slots: int) -> float:
        """Return target launch duration for the rectangular verifier batch."""

        if (
            isinstance(verifier_slots, bool)
            or not isinstance(verifier_slots, int)
            or verifier_slots < batch_rows
        ):
            raise ValueError("verifier_slots must be at least batch_rows")
        return self.target_curve(batch_rows) + verifier_slots * self.verifier_slot_ms

    def draft_latency_ms(self, batch_rows: int, *, recovery: bool = False) -> float:
        """Return draft-engine duration for preparation or cache recovery."""

        curve = self.recovery_curve if recovery else self.draft_curve
        return curve(batch_rows)
