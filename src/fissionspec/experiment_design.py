"""Executable experiment-design rules shared by CPU and GPU studies."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from .statistics import ConfidenceSequencePoint, bounded_mean_confidence_sequence

MetricDirection = Literal["higher", "lower"]


def symmetric_improvement(
    candidate: float,
    baseline: float,
    *,
    direction: MetricDirection,
) -> float:
    """Return a bounded paired improvement for non-negative metrics.

    The result is in ``[-1, 1]`` and treats equal zeros as no difference. This
    estimand avoids unbounded ratios in predeclared sequential GPU inference.
    """

    if direction not in {"higher", "lower"}:
        raise ValueError("direction must be 'higher' or 'lower'")
    for field, value in (("candidate", candidate), ("baseline", baseline)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0.0
        ):
            raise ValueError(f"{field} must be finite and non-negative")
    candidate_value = float(candidate)
    baseline_value = float(baseline)
    scale = max(candidate_value, baseline_value)
    if scale == 0.0:
        return 0.0
    orientation = 1.0 if direction == "higher" else -1.0
    result = orientation * (candidate_value - baseline_value) / scale
    return min(1.0, max(-1.0, result))


def paired_block_order(block_index: int) -> tuple[str, str, str, str]:
    """Return the frozen alternating ABBA/BAAB policy order."""

    if isinstance(block_index, bool) or not isinstance(block_index, int) or block_index < 0:
        raise ValueError("block_index must be a non-negative integer")
    if block_index % 2 == 0:
        return ("candidate", "baseline", "baseline", "candidate")
    return ("baseline", "candidate", "candidate", "baseline")


@dataclass(frozen=True, slots=True)
class SequentialGateConfig:
    """Predeclared stopping parameters for one paired metric."""

    minimum_blocks: int = 10
    maximum_blocks: int = 50
    look_every: int = 5
    confidence_level: float = 0.95
    target_half_width: float = 0.03
    minimum_worthwhile_improvement: float = 0.03

    def __post_init__(self) -> None:
        for integer_field, integer_value in (
            ("minimum_blocks", self.minimum_blocks),
            ("maximum_blocks", self.maximum_blocks),
            ("look_every", self.look_every),
        ):
            if (
                isinstance(integer_value, bool)
                or not isinstance(integer_value, int)
                or integer_value <= 0
            ):
                raise ValueError(f"{integer_field} must be a positive integer")
        if self.minimum_blocks > self.maximum_blocks:
            raise ValueError("minimum_blocks must not exceed maximum_blocks")
        if self.minimum_blocks % self.look_every != 0:
            raise ValueError("minimum_blocks must fall on a scheduled look")
        if self.maximum_blocks % self.look_every != 0:
            raise ValueError("maximum_blocks must fall on a scheduled look")
        if (
            isinstance(self.confidence_level, bool)
            or not isinstance(self.confidence_level, (int, float))
            or not math.isfinite(self.confidence_level)
            or not 0.0 < self.confidence_level < 1.0
        ):
            raise ValueError("confidence_level must be strictly between zero and one")
        for float_field, float_value in (
            ("target_half_width", self.target_half_width),
            (
                "minimum_worthwhile_improvement",
                self.minimum_worthwhile_improvement,
            ),
        ):
            if (
                isinstance(float_value, bool)
                or not isinstance(float_value, (int, float))
                or not math.isfinite(float_value)
                or float_value <= 0.0
                or float_value > 1.0
            ):
                raise ValueError(f"{float_field} must be finite and in (0, 1]")


class GateStatus(StrEnum):
    """Decision at one predeclared sequential look."""

    NOT_A_LOOK = "not_a_look"
    CONTINUE = "continue"
    EFFICACY = "efficacy"
    FUTILITY = "futility"
    POSITIVE_BELOW_MWI = "positive_below_minimum_worthwhile_improvement"
    PRECISE_INCONCLUSIVE = "precise_inconclusive"
    MAXIMUM_REACHED = "maximum_reached"


@dataclass(frozen=True, slots=True)
class SequentialGateDecision:
    """Auditable outcome from a predeclared sequential gate."""

    status: GateStatus
    terminal: bool
    blocks: int
    interval: ConfidenceSequencePoint | None
    rule: str


def evaluate_sequential_gate(
    improvements: Sequence[float],
    config: SequentialGateConfig | None = None,
) -> SequentialGateDecision:
    """Evaluate the current look without consuming or discarding observations."""

    if config is None:
        config = SequentialGateConfig()
    values = tuple(float(value) for value in improvements)
    if not values:
        return SequentialGateDecision(
            GateStatus.NOT_A_LOOK,
            terminal=False,
            blocks=0,
            interval=None,
            rule="minimum independent paired blocks not reached",
        )
    if len(values) > config.maximum_blocks:
        raise ValueError("observations exceed the predeclared maximum_blocks")
    if any(not math.isfinite(value) or not -1.0 <= value <= 1.0 for value in values):
        raise ValueError("improvements must be finite and in [-1, 1]")
    if len(values) < config.minimum_blocks or len(values) % config.look_every != 0:
        return SequentialGateDecision(
            GateStatus.NOT_A_LOOK,
            terminal=False,
            blocks=len(values),
            interval=None,
            rule="evaluate only at predeclared completed-block looks",
        )
    interval = bounded_mean_confidence_sequence(
        values,
        lower_bound=-1.0,
        upper_bound=1.0,
        confidence_level=config.confidence_level,
    )[-1]
    positive = interval.lower > 0.0
    below_worthwhile = interval.upper < config.minimum_worthwhile_improvement
    if positive and below_worthwhile:
        return SequentialGateDecision(
            GateStatus.POSITIVE_BELOW_MWI,
            terminal=True,
            blocks=len(values),
            interval=interval,
            rule=(
                "effect is statistically positive but its entire interval is "
                "below the minimum worthwhile improvement"
            ),
        )
    if below_worthwhile:
        return SequentialGateDecision(
            GateStatus.FUTILITY,
            terminal=True,
            blocks=len(values),
            interval=interval,
            rule="upper confidence bound is below the minimum worthwhile improvement",
        )
    if positive:
        return SequentialGateDecision(
            GateStatus.EFFICACY,
            terminal=True,
            blocks=len(values),
            interval=interval,
            rule="lower confidence bound exceeds zero",
        )
    if interval.half_width <= config.target_half_width:
        return SequentialGateDecision(
            GateStatus.PRECISE_INCONCLUSIVE,
            terminal=True,
            blocks=len(values),
            interval=interval,
            rule="precision target reached without efficacy or futility",
        )
    if len(values) == config.maximum_blocks:
        return SequentialGateDecision(
            GateStatus.MAXIMUM_REACHED,
            terminal=True,
            blocks=len(values),
            interval=interval,
            rule="predeclared maximum independent paired blocks reached",
        )
    return SequentialGateDecision(
        GateStatus.CONTINUE,
        terminal=False,
        blocks=len(values),
        interval=interval,
        rule="no terminal boundary crossed",
    )


@dataclass(frozen=True, slots=True)
class CalibrationRefinement:
    """Deterministic result of the anchor interpolation check."""

    width: int
    observed_middle_ms: float
    predicted_middle_ms: float
    relative_error: float
    add_batch_rows: tuple[int, ...]


def calibration_refinement_plan(
    anchor_measurements: Mapping[tuple[int, int], float],
    *,
    anchor_rows: tuple[int, int, int] = (1, 8, 32),
    intermediate_rows: tuple[int, ...] = (2, 4, 16),
    relative_error_threshold: float = 0.03,
) -> tuple[CalibrationRefinement, ...]:
    """Apply the frozen leave-middle-anchor-out refinement rule per width."""

    low, middle, high = anchor_rows
    if not 0 < low < middle < high:
        raise ValueError("anchor_rows must contain three strictly increasing positives")
    if any(
        isinstance(row, bool) or not isinstance(row, int) or row <= 0 for row in intermediate_rows
    ):
        raise ValueError("intermediate_rows must be positive integers")
    if (
        isinstance(relative_error_threshold, bool)
        or not isinstance(relative_error_threshold, (int, float))
        or not math.isfinite(relative_error_threshold)
        or relative_error_threshold < 0.0
    ):
        raise ValueError("relative_error_threshold must be finite and non-negative")
    widths = sorted({width for _, width in anchor_measurements})
    if not widths:
        raise ValueError("anchor_measurements must not be empty")
    results: list[CalibrationRefinement] = []
    fraction = (middle - low) / (high - low)
    for width in widths:
        values: list[float] = []
        for row in anchor_rows:
            raw = anchor_measurements.get((row, width))
            if (
                raw is None
                or isinstance(raw, bool)
                or not isinstance(raw, (int, float))
                or not math.isfinite(raw)
                or raw <= 0.0
            ):
                raise ValueError(
                    f"missing finite positive anchor measurement for rows={row}, width={width}"
                )
            values.append(float(raw))
        predicted = values[0] + fraction * (values[2] - values[0])
        error = abs(predicted - values[1]) / values[1]
        additions = intermediate_rows if error > relative_error_threshold else ()
        results.append(
            CalibrationRefinement(
                width=width,
                observed_middle_ms=values[1],
                predicted_middle_ms=predicted,
                relative_error=error,
                add_batch_rows=additions,
            )
        )
    return tuple(results)


@dataclass(frozen=True, slots=True)
class DesignCell:
    """One normalized controller-phase candidate for deterministic selection."""

    cell_id: str
    region: str
    parameters: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.cell_id or not self.region:
            raise ValueError("cell_id and region must not be empty")
        if not self.parameters or any(not math.isfinite(value) for value in self.parameters):
            raise ValueError("parameters must contain finite values")


def _normalized_parameters(cells: Sequence[DesignCell]) -> dict[str, tuple[float, ...]]:
    dimension = len(cells[0].parameters)
    if any(len(cell.parameters) != dimension for cell in cells):
        raise ValueError("all design cells must have equal parameter dimension")
    minima = tuple(min(cell.parameters[index] for cell in cells) for index in range(dimension))
    maxima = tuple(max(cell.parameters[index] for cell in cells) for index in range(dimension))
    return {
        cell.cell_id: tuple(
            (
                0.0
                if maxima[index] == minima[index]
                else (value - minima[index]) / (maxima[index] - minima[index])
            )
            for index, value in enumerate(cell.parameters)
        )
        for cell in cells
    }


def select_farthest_cells(
    cells: Sequence[DesignCell],
    *,
    per_region: Mapping[str, int],
) -> tuple[DesignCell, ...]:
    """Select deterministic farthest-point coverage within every region."""

    candidates = tuple(cells)
    if not candidates:
        raise ValueError("cells must not be empty")
    if len({cell.cell_id for cell in candidates}) != len(candidates):
        raise ValueError("cell_id values must be unique")
    for region, count in per_region.items():
        if not region or isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("per_region must map non-empty regions to positive counts")
    normalized = _normalized_parameters(candidates)
    selected: list[DesignCell] = []
    for region in sorted(per_region):
        pool = sorted(
            (cell for cell in candidates if cell.region == region),
            key=lambda cell: cell.cell_id,
        )
        required = per_region[region]
        if len(pool) < required:
            raise ValueError(f"region {region!r} has {len(pool)} cells but needs {required}")
        chosen = [pool.pop(0)]
        while len(chosen) < required:
            distances = {
                cell.cell_id: min(
                    math.dist(
                        normalized[cell.cell_id],
                        normalized[other.cell_id],
                    )
                    for other in chosen
                )
                for cell in pool
            }
            best_distance = max(distances.values())
            best = min(
                (
                    cell
                    for cell in pool
                    if math.isclose(
                        distances[cell.cell_id],
                        best_distance,
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    )
                ),
                key=lambda cell: cell.cell_id,
            )
            chosen.append(best)
            pool.remove(best)
        selected.extend(chosen)
    return tuple(selected)


__all__ = [
    "CalibrationRefinement",
    "DesignCell",
    "GateStatus",
    "MetricDirection",
    "SequentialGateConfig",
    "SequentialGateDecision",
    "calibration_refinement_plan",
    "evaluate_sequential_gate",
    "paired_block_order",
    "select_farthest_cells",
    "symmetric_improvement",
]
