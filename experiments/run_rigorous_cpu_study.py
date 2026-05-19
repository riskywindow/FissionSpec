#!/usr/bin/env python3
"""Analyze paired per-seed simulator rows with explicit statistical controls.

This driver consumes an existing per-seed CSV. It never runs a GPU benchmark
and does not modify the checked-in synthetic sweep.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from fissionspec.statistics import (
    Direction,
    bonferroni_metadata,
    paired_cluster_bootstrap,
    paired_effect_size,
    paired_replication_plan,
    precision_stopping,
)

SCHEMA_VERSION: Final = 1
WARNING: Final = "SYNTHETIC CPU MODEL STATISTICAL ANALYSIS — NOT GPU MEASUREMENTS."
CLAIM_BOUNDARY: Final = (
    "Intervals quantify variation across paired simulator seed clusters only; "
    "they are not hardware-performance confidence intervals."
)
_INTEGER_SEED = re.compile(r"^[+-]?[0-9]+$")


class StudyError(ValueError):
    """Raised when an input table cannot support a valid paired analysis."""


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """One analyzed metric and its improvement direction."""

    name: str
    direction: Direction

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("metric name must be a non-empty string")
        if self.direction not in {"higher", "lower"}:
            raise ValueError("metric direction must be 'higher' or 'lower'")


@dataclass(frozen=True, slots=True)
class PrecisionSpec:
    """Predeclared bounds for anytime-valid oriented paired differences."""

    metric: str
    lower_difference: float
    upper_difference: float
    target_half_width: float
    minimum_observations: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.metric, str) or not self.metric:
            raise ValueError("precision metric must be a non-empty string")
        if (
            not math.isfinite(self.lower_difference)
            or not math.isfinite(self.upper_difference)
            or self.lower_difference > self.upper_difference
        ):
            raise ValueError("precision bounds must be finite and ordered")
        if not math.isfinite(self.target_half_width) or self.target_half_width <= 0.0:
            raise ValueError("precision target_half_width must be finite and positive")
        if (
            isinstance(self.minimum_observations, bool)
            or not isinstance(self.minimum_observations, int)
            or self.minimum_observations <= 0
        ):
            raise ValueError("minimum_observations must be a positive integer")


@dataclass(frozen=True, slots=True)
class StudyConfig:
    """Complete, immutable analysis declaration."""

    candidate_policy: str
    baseline_policy: str
    metrics: tuple[MetricSpec, ...]
    resamples: int = 10_000
    bootstrap_seed: str = "fissionspec-rigorous-cpu-study-v1"
    familywise_alpha: float = 0.05
    target_power: float = 0.8
    minimum_detectable_standardized_effect: float = 0.5
    precision_specs: tuple[PrecisionSpec, ...] = ()
    family_id: str = "all-workload-regime-metric-policy-comparisons"

    def __post_init__(self) -> None:
        for field_name in ("candidate_policy", "baseline_policy", "bootstrap_seed", "family_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a non-empty string")
        if self.candidate_policy == self.baseline_policy:
            raise ValueError("candidate and baseline policies must differ")
        if not self.metrics:
            raise ValueError("at least one metric must be declared")
        metric_names = tuple(metric.name for metric in self.metrics)
        if len(metric_names) != len(set(metric_names)):
            raise ValueError("metric names must be unique")
        if isinstance(self.resamples, bool) or not isinstance(self.resamples, int):
            raise ValueError("resamples must be an integer")
        if self.resamples < 100:
            raise ValueError("resamples must be at least 100")
        if not math.isfinite(self.familywise_alpha) or not 0.0 < self.familywise_alpha < 1.0:
            raise ValueError("familywise_alpha must be strictly between zero and one")
        if not math.isfinite(self.target_power) or not 0.5 < self.target_power < 1.0:
            raise ValueError("target_power must be strictly between 0.5 and one")
        if (
            not math.isfinite(self.minimum_detectable_standardized_effect)
            or self.minimum_detectable_standardized_effect <= 0.0
        ):
            raise ValueError("minimum detectable standardized effect must be positive")
        precision_metrics = tuple(spec.metric for spec in self.precision_specs)
        if len(precision_metrics) != len(set(precision_metrics)):
            raise ValueError("precision specs must name unique metrics")
        unknown_precision = set(precision_metrics) - set(metric_names)
        if unknown_precision:
            raise ValueError(
                f"precision specs reference undeclared metrics: {sorted(unknown_precision)}"
            )


@dataclass(frozen=True, slots=True)
class _InputRow:
    workload: str
    regime: str
    seed: str
    policy: str
    metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class _LoadedTable:
    rows: tuple[_InputRow, ...]
    row_count: int
    evidence_classes: tuple[str, ...]
    source_warnings: tuple[str, ...]


def _required_cell(row: dict[str, str | None], field: str, *, line_number: int) -> str:
    value = row.get(field)
    if value is None or not value.strip():
        raise StudyError(f"line {line_number}: missing {field}")
    return value.strip()


def _load_table(path: Path, config: StudyConfig) -> _LoadedTable:
    rows: list[_InputRow] = []
    evidence_classes: set[str] = set()
    source_warnings: set[str] = set()
    input_row_count = 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "evidence_class",
            "measurement_warning",
            "workload",
            "regime",
            "seed",
            "policy",
            *(metric.name for metric in config.metrics),
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or ()))
            raise StudyError(f"input CSV is missing columns: {missing}")
        for line_number, row in enumerate(reader, start=2):
            input_row_count += 1
            evidence = _required_cell(row, "evidence_class", line_number=line_number)
            warning = _required_cell(row, "measurement_warning", line_number=line_number)
            if "SYNTHETIC" not in evidence.upper() or "NOT GPU" not in warning.upper():
                raise StudyError(
                    f"line {line_number}: rigorous CPU driver accepts only explicitly "
                    "synthetic, not-GPU rows"
                )
            evidence_classes.add(evidence)
            source_warnings.add(warning)
            policy = _required_cell(row, "policy", line_number=line_number)
            if policy not in {config.candidate_policy, config.baseline_policy}:
                continue
            metric_values: dict[str, float] = {}
            for metric in config.metrics:
                raw = _required_cell(row, metric.name, line_number=line_number)
                try:
                    value = float(raw)
                except ValueError as exc:
                    raise StudyError(f"line {line_number}: {metric.name} must be numeric") from exc
                if not math.isfinite(value):
                    raise StudyError(f"line {line_number}: {metric.name} must be finite")
                metric_values[metric.name] = value
            rows.append(
                _InputRow(
                    workload=_required_cell(row, "workload", line_number=line_number),
                    regime=_required_cell(row, "regime", line_number=line_number),
                    seed=_required_cell(row, "seed", line_number=line_number),
                    policy=policy,
                    metrics=metric_values,
                )
            )
    if not rows:
        raise StudyError("input contains no rows for the declared candidate and baseline")
    return _LoadedTable(
        rows=tuple(rows),
        row_count=input_row_count,
        evidence_classes=tuple(sorted(evidence_classes)),
        source_warnings=tuple(sorted(source_warnings)),
    )


def _seed_sort_key(seed: str) -> tuple[int, int, str]:
    if _INTEGER_SEED.fullmatch(seed):
        return (0, int(seed), "")
    return (1, 0, seed)


def _paired_cells(
    table: _LoadedTable,
    config: StudyConfig,
) -> dict[tuple[str, str], dict[str, tuple[_InputRow, _InputRow]]]:
    indexed: dict[tuple[str, str, str, str], _InputRow] = {}
    for row in table.rows:
        key = (row.workload, row.regime, row.seed, row.policy)
        if key in indexed:
            raise StudyError(f"duplicate policy row for {key}")
        indexed[key] = row
    cells: dict[tuple[str, str], dict[str, tuple[_InputRow, _InputRow]]] = {}
    cell_keys = sorted({(row.workload, row.regime) for row in table.rows})
    for workload, regime in cell_keys:
        candidate_seeds = {
            row.seed
            for row in table.rows
            if row.workload == workload
            and row.regime == regime
            and row.policy == config.candidate_policy
        }
        baseline_seeds = {
            row.seed
            for row in table.rows
            if row.workload == workload
            and row.regime == regime
            and row.policy == config.baseline_policy
        }
        if candidate_seeds != baseline_seeds:
            raise StudyError(
                f"unpaired seeds for workload={workload!r}, regime={regime!r}: "
                f"candidate_only={sorted(candidate_seeds - baseline_seeds)}, "
                f"baseline_only={sorted(baseline_seeds - candidate_seeds)}"
            )
        if len(candidate_seeds) < 2:
            raise StudyError(
                f"workload={workload!r}, regime={regime!r} needs at least two seed clusters"
            )
        pairs: dict[str, tuple[_InputRow, _InputRow]] = {}
        for seed in sorted(candidate_seeds, key=_seed_sort_key):
            pairs[seed] = (
                indexed[(workload, regime, seed, config.candidate_policy)],
                indexed[(workload, regime, seed, config.baseline_policy)],
            )
        cells[(workload, regime)] = pairs
    return cells


def _hypothesis_id(
    workload: str,
    regime: str,
    metric: str,
    config: StudyConfig,
) -> str:
    return f"{workload}/{regime}/{metric}/{config.candidate_policy}-vs-{config.baseline_policy}"


def analyze_csv(path: Path, config: StudyConfig) -> dict[str, object]:
    """Analyze one strict per-seed CSV and return a deterministic evidence document."""

    serialized = path.read_bytes()
    table = _load_table(path, config)
    cells = _paired_cells(table, config)
    hypothesis_ids = tuple(
        _hypothesis_id(workload, regime, metric.name, config)
        for workload, regime in sorted(cells)
        for metric in config.metrics
    )
    multiplicity = bonferroni_metadata(
        hypothesis_ids,
        family_id=config.family_id,
        familywise_alpha=config.familywise_alpha,
        confirmatory=False,
    )
    precision_by_metric = {spec.metric: spec for spec in config.precision_specs}
    comparisons: list[dict[str, object]] = []
    for workload, regime in sorted(cells):
        pairs = cells[(workload, regime)]
        ordered_seeds = tuple(pairs)
        for metric in config.metrics:
            candidate_values = tuple(pairs[seed][0].metrics[metric.name] for seed in ordered_seeds)
            baseline_values = tuple(pairs[seed][1].metrics[metric.name] for seed in ordered_seeds)
            orientation = 1.0 if metric.direction == "higher" else -1.0
            oriented_differences = tuple(
                orientation * (candidate - baseline)
                for candidate, baseline in zip(
                    candidate_values,
                    baseline_values,
                    strict=True,
                )
            )
            effects = paired_effect_size(
                candidate_values,
                baseline_values,
                direction=metric.direction,
            )
            interval = paired_cluster_bootstrap(
                {
                    seed: (difference,)
                    for seed, difference in zip(
                        ordered_seeds,
                        oriented_differences,
                        strict=True,
                    )
                },
                confidence_level=(multiplicity.simultaneous_per_hypothesis_confidence_level),
                resamples=config.resamples,
                seed=config.bootstrap_seed,
            )
            replication = paired_replication_plan(
                oriented_differences,
                minimum_detectable_standardized_effect=(
                    config.minimum_detectable_standardized_effect
                ),
                familywise_alpha=config.familywise_alpha,
                target_power=config.target_power,
                hypotheses=len(hypothesis_ids),
            )
            precision_spec = precision_by_metric.get(metric.name)
            if precision_spec is None:
                precision: dict[str, object] = {
                    "status": "not-run",
                    "reason": (
                        "anytime-valid stopping requires predeclared finite bounds "
                        "on oriented paired differences"
                    ),
                }
            else:
                precision_result = precision_stopping(
                    oriented_differences,
                    lower_bound=precision_spec.lower_difference,
                    upper_bound=precision_spec.upper_difference,
                    target_half_width=precision_spec.target_half_width,
                    minimum_observations=precision_spec.minimum_observations,
                    confidence_level=(multiplicity.simultaneous_per_hypothesis_confidence_level),
                )
                precision = {"status": "run", **precision_result.as_dict()}
            comparisons.append(
                {
                    "hypothesis_id": _hypothesis_id(
                        workload,
                        regime,
                        metric.name,
                        config,
                    ),
                    "workload": workload,
                    "regime": regime,
                    "metric": metric.name,
                    "improvement_direction": metric.direction,
                    "pairing_unit": "seed",
                    "cluster_ids": list(ordered_seeds),
                    "independent_clusters": len(ordered_seeds),
                    "effect": effects.as_dict(),
                    "oriented_improvement_simultaneous_interval": interval.as_dict(),
                    "precision_stopping": precision,
                    "replication_plan": replication.as_dict(),
                    "interpretation_warning": (
                        "fewer than 10 independent seed clusters; treat effect and "
                        "interval as pilot evidence"
                        if len(ordered_seeds) < 10
                        else None
                    ),
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": "FissionSpec rigorous paired CPU study",
        "evidence_class": "synthetic-cpu-model-statistics",
        "measurement_warning": WARNING,
        "claim_boundary": CLAIM_BOUNDARY,
        "inference_status": "exploratory-not-confirmatory",
        "input": {
            "path": str(path),
            "sha256": hashlib.sha256(serialized).hexdigest(),
            "row_count": table.row_count,
            "evidence_classes": list(table.evidence_classes),
            "source_measurement_warnings": list(table.source_warnings),
        },
        "design": {
            "candidate_policy": config.candidate_policy,
            "baseline_policy": config.baseline_policy,
            "paired_unit": "seed within workload and regime",
            "independent_resampling_unit": "seed cluster",
            "metrics": [
                {"name": metric.name, "improvement_direction": metric.direction}
                for metric in config.metrics
            ],
            "bootstrap": {
                "method": "paired percentile cluster bootstrap",
                "resamples": config.resamples,
                "seed": config.bootstrap_seed,
                "common_resample_schedule_across_hypotheses": True,
            },
            "multiplicity": multiplicity.as_dict(),
            "precision_stopping": {
                "method": "alpha-spending Hoeffding confidence sequence",
                "bounds_source": "must be predeclared in StudyConfig",
                "specifications": [
                    {
                        "metric": spec.metric,
                        "oriented_difference_bounds": [
                            spec.lower_difference,
                            spec.upper_difference,
                        ],
                        "target_half_width": spec.target_half_width,
                        "minimum_observations": spec.minimum_observations,
                    }
                    for spec in config.precision_specs
                ],
            },
            "power_planning": {
                "target_power": config.target_power,
                "minimum_detectable_standardized_effect": (
                    config.minimum_detectable_standardized_effect
                ),
                "familywise_alpha": config.familywise_alpha,
                "hypotheses": len(hypothesis_ids),
            },
        },
        "comparisons": comparisons,
    }


def write_document(path: Path, document: dict[str, object], *, force: bool = False) -> None:
    """Write a stable JSON document, refusing accidental overwrite by default."""

    if path.exists() and not force:
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _metric_argument(raw: str) -> MetricSpec:
    try:
        name, direction = raw.rsplit(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("metric must be NAME:higher or NAME:lower") from exc
    try:
        return MetricSpec(name=name, direction=direction)  # type: ignore[arg-type]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _precision_argument(raw: str) -> PrecisionSpec:
    parts = raw.split(":")
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            "precision must be METRIC:LOWER_DIFF:UPPER_DIFF:HALF_WIDTH:MIN_N"
        )
    metric, lower, upper, half_width, minimum = parts
    try:
        return PrecisionSpec(
            metric=metric,
            lower_difference=float(lower),
            upper_difference=float(upper),
            target_half_width=float(half_width),
            minimum_observations=int(minimum),
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "synthetic_sweep.csv",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", default="fissionspec-horizon-2")
    parser.add_argument("--baseline", default="saguaro-barrier")
    parser.add_argument(
        "--metric",
        action="append",
        type=_metric_argument,
        help="repeat NAME:higher|lower; defaults to throughput, P95 TBT, and request SLO",
    )
    parser.add_argument(
        "--precision",
        action="append",
        type=_precision_argument,
        default=[],
        help="repeat METRIC:LOWER_DIFF:UPPER_DIFF:HALF_WIDTH:MIN_N",
    )
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", default="fissionspec-rigorous-cpu-study-v1")
    parser.add_argument("--familywise-alpha", type=float, default=0.05)
    parser.add_argument("--target-power", type=float, default=0.8)
    parser.add_argument("--minimum-standardized-effect", type=float, default=0.5)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    metrics = (
        tuple(args.metric)
        if args.metric
        else (
            MetricSpec("throughput_tokens_per_s", "higher"),
            MetricSpec("p95_tbt_ms", "lower"),
            MetricSpec("request_tbt_slo_attainment", "higher"),
        )
    )
    config = StudyConfig(
        candidate_policy=args.candidate,
        baseline_policy=args.baseline,
        metrics=metrics,
        resamples=args.resamples,
        bootstrap_seed=args.bootstrap_seed,
        familywise_alpha=args.familywise_alpha,
        target_power=args.target_power,
        minimum_detectable_standardized_effect=args.minimum_standardized_effect,
        precision_specs=tuple(args.precision),
    )
    document = analyze_csv(args.input, config)
    write_document(args.output, document, force=args.force)
    print(WARNING)
    print(CLAIM_BOUNDARY)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
