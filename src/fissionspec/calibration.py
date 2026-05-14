"""Robust conversion of raw engine timings into simulator latency profiles.

The fitter intentionally uses only transparent statistics.  Repeated samples
are reduced with medians, the target token-axis coefficient is estimated with a
within-row Theil--Sen slope, and remaining batch curves are projected onto the
monotone cone with weighted pool-adjacent-violators regression.  The result is
therefore inspectable and deterministic rather than an opaque learned cost
model.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

Component = Literal["target", "draft", "recovery"]
_COMPONENTS: tuple[Component, ...] = ("target", "draft", "recovery")


class CalibrationError(ValueError):
    """Raised when timing samples cannot identify a valid profile."""


@dataclass(frozen=True, slots=True)
class TimingSample:
    """One post-warmup latency observation from a serving engine."""

    component: Component
    batch_rows: int
    latency_ms: float
    verifier_slots: int = 0

    def __post_init__(self) -> None:
        if self.component not in _COMPONENTS:
            raise CalibrationError(f"unknown component: {self.component!r}")
        if (
            isinstance(self.batch_rows, bool)
            or not isinstance(self.batch_rows, int)
            or self.batch_rows <= 0
        ):
            raise CalibrationError("batch_rows must be a positive integer")
        if (
            isinstance(self.latency_ms, bool)
            or not isinstance(self.latency_ms, (int, float))
            or not math.isfinite(self.latency_ms)
            or self.latency_ms <= 0.0
        ):
            raise CalibrationError("latency_ms must be finite and positive")
        if (
            isinstance(self.verifier_slots, bool)
            or not isinstance(self.verifier_slots, int)
            or self.verifier_slots < 0
        ):
            raise CalibrationError("verifier_slots must be a non-negative integer")
        if self.component == "target" and self.verifier_slots < self.batch_rows:
            raise CalibrationError(
                "target verifier_slots must be at least the number of batch rows"
            )
        if self.component != "target" and self.verifier_slots != 0:
            raise CalibrationError(
                "draft and recovery samples must set verifier_slots to zero"
            )


@dataclass(frozen=True, slots=True)
class FittedProfile:
    """A fitted profile plus diagnostics that should accompany publications."""

    name: str
    target_curve: tuple[tuple[int, float], ...]
    draft_curve: tuple[tuple[int, float], ...]
    recovery_curve: tuple[tuple[int, float], ...]
    verifier_slot_ms: float
    raw_verifier_slot_ms: float
    sample_count: int
    target_rmse_ms: float
    slot_slope_identified: bool
    slot_slope_clipped: bool

    def as_dict(self, provenance: dict[str, object] | None = None) -> dict[str, object]:
        effective_provenance = (
            provenance
            if provenance is not None
            else {
                "kind": "unverified-measurement",
                "publication_ready": False,
                "warning": (
                    "No engine/GPU/model provenance was supplied. "
                    "Do not use this profile for performance claims."
                ),
            }
        )
        return {
            "schema_version": 1,
            "name": self.name,
            "provenance": effective_provenance,
            "fit": {
                "method": "median+within-row-theil-sen+weighted-pava",
                "sample_count": self.sample_count,
                "target_rmse_ms": self.target_rmse_ms,
                "slot_slope_identified": self.slot_slope_identified,
                "slot_slope_clipped": self.slot_slope_clipped,
                "raw_verifier_slot_ms": self.raw_verifier_slot_ms,
            },
            "target_curve": [list(point) for point in self.target_curve],
            "draft_curve": [list(point) for point in self.draft_curve],
            "recovery_curve": [list(point) for point in self.recovery_curve],
            "verifier_slot_ms": self.verifier_slot_ms,
        }


@dataclass(slots=True)
class _PavaBlock:
    """One mutable block in weighted pool-adjacent-violators regression."""

    start: int
    end: int
    weighted_sum: float
    weight: int

    @property
    def mean(self) -> float:
        return self.weighted_sum / self.weight


def _weighted_pava(
    points: Sequence[tuple[int, float, int]],
) -> tuple[tuple[int, float], ...]:
    """Return the weighted least-squares nondecreasing projection.

    Each input tuple is ``(batch_rows, value, replicate_count)`` and batch rows
    must already be unique.  Adjacent violating blocks are pooled until their
    weighted means are monotone.
    """

    if not points:
        raise CalibrationError("cannot fit an empty latency curve")
    ordered = sorted(points)
    if len({row for row, _, _ in ordered}) != len(ordered):
        raise CalibrationError("PAVA points must have unique batch sizes")

    blocks: list[_PavaBlock] = []
    for index, (row, value, weight) in enumerate(ordered):
        if (
            isinstance(row, bool)
            or not isinstance(row, int)
            or row <= 0
            or isinstance(weight, bool)
            or not isinstance(weight, int)
            or weight <= 0
            or not math.isfinite(value)
        ):
            raise CalibrationError("PAVA rows, values, and weights must be valid")
        blocks.append(_PavaBlock(index, index, value * weight, weight))
        while len(blocks) >= 2:
            left, right = blocks[-2], blocks[-1]
            if left.mean <= right.mean:
                break
            blocks[-2:] = [
                _PavaBlock(
                    left.start,
                    right.end,
                    left.weighted_sum + right.weighted_sum,
                    left.weight + right.weight,
                )
            ]

    fitted = [0.0] * len(ordered)
    for block in blocks:
        if block.mean <= 0.0:
            raise CalibrationError(
                "fitted latency baseline is non-positive; collect a wider, lower-noise slot sweep"
            )
        for index in range(block.start, block.end + 1):
            fitted[index] = block.mean
    return tuple((row, fitted[index]) for index, (row, _, _) in enumerate(ordered))


def _group(samples: Iterable[TimingSample]) -> dict[int, list[TimingSample]]:
    grouped: dict[int, list[TimingSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.batch_rows, []).append(sample)
    return grouped


def _fit_plain_curve(samples: Sequence[TimingSample]) -> tuple[tuple[int, float], ...]:
    grouped = _group(samples)
    points = [
        (row, statistics.median(sample.latency_ms for sample in group), len(group))
        for row, group in grouped.items()
    ]
    return _weighted_pava(points)


def _target_slot_slope(
    groups: dict[int, list[TimingSample]],
) -> tuple[float, float, bool]:
    """Estimate token-axis cost without confounding it with row count.

    Replicates are first collapsed to a median for each ``(rows, slots)`` cell.
    A Theil--Sen slope is then computed independently inside each row group,
    and the row estimates are combined by their median.  Equal row weighting
    prevents a densely sampled batch size from dominating the token-axis
    coefficient.
    """

    row_slopes: list[float] = []
    for group in groups.values():
        by_slots: dict[int, list[float]] = {}
        for sample in group:
            by_slots.setdefault(sample.verifier_slots, []).append(sample.latency_ms)
        slot_medians = sorted(
            (slots, statistics.median(latencies)) for slots, latencies in by_slots.items()
        )
        slopes: list[float] = []
        for left_index, (left_slots, left_latency) in enumerate(slot_medians):
            for right_slots, right_latency in slot_medians[left_index + 1 :]:
                slopes.append((right_latency - left_latency) / (right_slots - left_slots))
        if slopes:
            row_slopes.append(statistics.median(slopes))
    if not row_slopes:
        return 0.0, 0.0, False
    raw_slope = statistics.median(row_slopes)
    return max(0.0, raw_slope), raw_slope, True


def fit_profile(samples: Iterable[TimingSample], *, name: str = "calibrated") -> FittedProfile:
    """Fit a monotone hardware profile from repeated raw measurements."""

    materialized = tuple(samples)
    if not materialized:
        raise CalibrationError("at least one timing sample is required")
    if not isinstance(name, str) or not name.strip():
        raise CalibrationError("profile name must not be empty")
    normalized_name = name.strip()
    by_component = {
        component: tuple(sample for sample in materialized if sample.component == component)
        for component in _COMPONENTS
    }
    missing = [component for component, values in by_component.items() if not values]
    if missing:
        raise CalibrationError(f"missing timing components: {', '.join(missing)}")

    target_groups = _group(by_component["target"])
    slot_slope, raw_slot_slope, identified = _target_slot_slope(target_groups)
    target_points: list[tuple[int, float, int]] = []
    for row, group in target_groups.items():
        bases = [sample.latency_ms - slot_slope * sample.verifier_slots for sample in group]
        target_points.append((row, statistics.median(bases), len(group)))
    target_curve = _weighted_pava(target_points)
    target_by_row = dict(target_curve)
    squared_error = sum(
        (
            sample.latency_ms
            - (target_by_row[sample.batch_rows] + slot_slope * sample.verifier_slots)
        )
        ** 2
        for sample in by_component["target"]
    )
    target_rmse = math.sqrt(squared_error / len(by_component["target"]))

    return FittedProfile(
        name=normalized_name,
        target_curve=target_curve,
        draft_curve=_fit_plain_curve(by_component["draft"]),
        recovery_curve=_fit_plain_curve(by_component["recovery"]),
        verifier_slot_ms=slot_slope,
        raw_verifier_slot_ms=raw_slot_slope,
        sample_count=len(materialized),
        target_rmse_ms=target_rmse,
        slot_slope_identified=identified,
        slot_slope_clipped=identified and raw_slot_slope < 0.0,
    )


def _csv_cell(row: Mapping[str, str | None], field: str, *, line_number: int) -> str:
    value = row.get(field)
    if value is None or not value.strip():
        raise CalibrationError(f"missing {field!r} value at CSV line {line_number}")
    return value.strip()


def load_samples_csv(path: Path) -> tuple[TimingSample, ...]:
    """Load the strict raw-trace CSV schema used by the calibration CLI."""

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = ("component", "batch_rows", "latency_ms", "verifier_slots")
        if (
            reader.fieldnames is None
            or len(reader.fieldnames) != len(required)
            or set(reader.fieldnames) != set(required)
        ):
            raise CalibrationError(
                "CSV columns must be exactly component,batch_rows,latency_ms,verifier_slots"
            )
        samples: list[TimingSample] = []
        for line_number, row in enumerate(reader, start=2):
            try:
                if None in row:
                    raise CalibrationError(
                        f"too many fields at CSV line {line_number}"
                    )
                raw_component = _csv_cell(row, "component", line_number=line_number)
                if raw_component not in _COMPONENTS:
                    raise CalibrationError(f"unknown component: {raw_component!r}")
                component = cast(Component, raw_component)
                samples.append(
                    TimingSample(
                        component=component,
                        batch_rows=int(_csv_cell(row, "batch_rows", line_number=line_number)),
                        latency_ms=float(_csv_cell(row, "latency_ms", line_number=line_number)),
                        verifier_slots=int(
                            _csv_cell(row, "verifier_slots", line_number=line_number)
                        ),
                    )
                )
            except (CalibrationError, TypeError, ValueError) as exc:
                raise CalibrationError(f"invalid sample at CSV line {line_number}: {exc}") from exc
    return tuple(samples)


def write_profile_json(
    profile: FittedProfile,
    output: Path,
    *,
    provenance: dict[str, object] | None = None,
) -> None:
    """Write a deterministic, simulator-compatible calibration document."""

    try:
        serialized = json.dumps(
            profile.as_dict(provenance),
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise CalibrationError(f"profile metadata is not valid JSON: {exc}") from exc
    output.write_text(serialized + "\n", encoding="utf-8")


__all__ = [
    "CalibrationError",
    "FittedProfile",
    "TimingSample",
    "fit_profile",
    "load_samples_csv",
    "write_profile_json",
]
