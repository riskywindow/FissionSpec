"""Executable experiment-design rules shared by CPU and GPU studies."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Final, Literal, cast

from .rng import CounterRNG, Seed
from .statistics import (
    FixedLookInterval,
    fixed_look_hoeffding_interval,
    fixed_look_student_t_interval,
    student_t_quantile,
)

MetricDirection = Literal["higher", "lower"]
DiagnosticDistribution = Literal["normal", "symmetric-two-point", "skewed-bounded"]

SEQUENTIAL_PROTOCOL_VERSION: Final = 2
PRIMARY_FAMILY_ID: Final = "gpu-primary-family-v2"
PRIMARY_MODEL_PAIRS: Final = ("qwen3-32b__qwen3-0.6b", "llama3.1-70b__llama3.2-1b")
PRIMARY_VALIDATION_ANCHORS: Final = ("v1-mmpp", "v2-pareto", "v3-replay")
PRIMARY_METRICS: Final = (
    "tbt-slo-goodput",
    "p99-tbt",
    "conditional-hit-delay",
    "one-miss-target-step-time",
)
PRIMARY_HYPOTHESIS_IDS: Final = tuple(
    f"{model_pair}/{anchor}/{metric}"
    for model_pair in PRIMARY_MODEL_PAIRS
    for anchor in PRIMARY_VALIDATION_ANCHORS
    for metric in PRIMARY_METRICS
)


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
    """Versioned finite-family, finite-look stopping declaration."""

    protocol_version: int = SEQUENTIAL_PROTOCOL_VERSION
    family_id: str = PRIMARY_FAMILY_ID
    hypothesis_ids: tuple[str, ...] = PRIMARY_HYPOTHESIS_IDS
    minimum_blocks: int = 10
    maximum_blocks: int = 50
    look_every: int = 5
    familywise_alpha: float = 0.05
    target_half_width: float = 0.03
    minimum_worthwhile_improvement: float = 0.03

    def __post_init__(self) -> None:
        if self.protocol_version != SEQUENTIAL_PROTOCOL_VERSION:
            raise ValueError(
                f"protocol_version must equal frozen version {SEQUENTIAL_PROTOCOL_VERSION}"
            )
        if not isinstance(self.family_id, str) or not self.family_id:
            raise ValueError("family_id must be a non-empty string")
        if (
            not self.hypothesis_ids
            or any(not isinstance(item, str) or not item for item in self.hypothesis_ids)
            or len(self.hypothesis_ids) != len(set(self.hypothesis_ids))
        ):
            raise ValueError("hypothesis_ids must be non-empty unique strings")
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
            isinstance(self.familywise_alpha, bool)
            or not isinstance(self.familywise_alpha, (int, float))
            or not math.isfinite(self.familywise_alpha)
            or not 0.0 < self.familywise_alpha < 1.0
        ):
            raise ValueError("familywise_alpha must be strictly between zero and one")
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

    @property
    def scheduled_looks(self) -> tuple[int, ...]:
        return tuple(
            range(
                self.minimum_blocks,
                self.maximum_blocks + 1,
                self.look_every,
            )
        )

    @property
    def per_interval_alpha(self) -> float:
        return self.familywise_alpha / (len(self.hypothesis_ids) * len(self.scheduled_looks))

    @property
    def simultaneous_interval_confidence_level(self) -> float:
        return 1.0 - self.per_interval_alpha

    def validate_observed_family(
        self,
        *,
        family_id: str,
        hypothesis_ids: Sequence[str],
    ) -> None:
        """Fail closed unless runtime family metadata exactly matches the protocol."""

        if family_id != self.family_id:
            raise ValueError("observed family_id does not match the frozen protocol")
        if tuple(hypothesis_ids) != self.hypothesis_ids:
            raise ValueError(
                "observed hypothesis_ids do not exactly match the frozen ordered family"
            )


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
    interval: FixedLookInterval | None
    distribution_free_sensitivity: FixedLookInterval | None
    rule: str
    protocol_version: int
    family_id: str
    hypothesis_id: str


@dataclass(frozen=True, slots=True)
class SequentialFamilyDecision:
    """One synchronized decision for the conjunctive 24-endpoint family."""

    status: GateStatus
    terminal: bool
    blocks: int
    endpoint_decisions: tuple[SequentialGateDecision, ...]
    rule: str
    protocol_version: int
    family_id: str


@dataclass(frozen=True, slots=True)
class ExperimentSpendCaps:
    """Hard replay-count caps for the registered accelerator campaign."""

    model_pairs: int = 2
    validation_anchors: int = 3
    minimum_blocks: int = 10
    maximum_blocks: int = 50
    runs_per_block: int = 4
    ablation_seeds: int = 10
    ablation_policies_including_candidate: int = 5
    robustness_cells: int = 12

    def __post_init__(self) -> None:
        for field_name in (
            "model_pairs",
            "validation_anchors",
            "minimum_blocks",
            "maximum_blocks",
            "runs_per_block",
            "ablation_seeds",
            "ablation_policies_including_candidate",
            "robustness_cells",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.minimum_blocks > self.maximum_blocks:
            raise ValueError("minimum_blocks must not exceed maximum_blocks")

    @property
    def minimum_primary_replays(self) -> int:
        return (
            self.model_pairs * self.validation_anchors * self.minimum_blocks * self.runs_per_block
        )

    @property
    def maximum_primary_replays(self) -> int:
        return (
            self.model_pairs * self.validation_anchors * self.maximum_blocks * self.runs_per_block
        )

    @property
    def maximum_unique_ablation_replays(self) -> int:
        return (
            self.model_pairs
            * self.validation_anchors
            * self.ablation_seeds
            * self.ablation_policies_including_candidate
        )

    def validate_manifest_counts(
        self,
        *,
        primary_replays: int,
        unique_ablation_replays: int,
        robustness_cells: int,
    ) -> None:
        """Reject a run manifest that can exceed any frozen spend boundary."""

        values = {
            "primary_replays": primary_replays,
            "unique_ablation_replays": unique_ablation_replays,
            "robustness_cells": robustness_cells,
        }
        for field_name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if primary_replays > self.maximum_primary_replays:
            raise ValueError("primary replay count exceeds the registered maximum")
        if unique_ablation_replays > self.maximum_unique_ablation_replays:
            raise ValueError("ablation replay count exceeds the registered maximum")
        if robustness_cells > self.robustness_cells:
            raise ValueError("robustness cell count exceeds the registered maximum")


def evaluate_sequential_gate(
    improvements: Sequence[float],
    config: SequentialGateConfig | None = None,
    *,
    hypothesis_id: str,
    observed_family_id: str,
    observed_hypothesis_ids: Sequence[str],
) -> SequentialGateDecision:
    """Evaluate one endpoint at a registered family look.

    The primary interval is a repeated-look Student-t interval with a
    Bonferroni allocation over every endpoint and look. Its finite-sample
    calibration assumes the paired-block Studentized mean follows the
    Student-t reference law. A bounded Hoeffding interval is always returned
    beside it as a distribution-free sensitivity analysis and never drives
    the primary stopping decision.
    """

    if config is None:
        config = SequentialGateConfig()
    config.validate_observed_family(
        family_id=observed_family_id,
        hypothesis_ids=observed_hypothesis_ids,
    )
    if hypothesis_id not in config.hypothesis_ids:
        raise ValueError("hypothesis_id is not in the frozen family")
    values = tuple(float(value) for value in improvements)
    if not values:
        return SequentialGateDecision(
            GateStatus.NOT_A_LOOK,
            terminal=False,
            blocks=0,
            interval=None,
            distribution_free_sensitivity=None,
            rule="minimum independent paired blocks not reached",
            protocol_version=config.protocol_version,
            family_id=config.family_id,
            hypothesis_id=hypothesis_id,
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
            distribution_free_sensitivity=None,
            rule="evaluate only at predeclared completed-block looks",
            protocol_version=config.protocol_version,
            family_id=config.family_id,
            hypothesis_id=hypothesis_id,
        )
    interval = fixed_look_student_t_interval(
        values,
        lower_bound=-1.0,
        upper_bound=1.0,
        familywise_alpha=config.familywise_alpha,
        hypotheses=len(config.hypothesis_ids),
        scheduled_looks=len(config.scheduled_looks),
    )
    sensitivity = fixed_look_hoeffding_interval(
        values,
        lower_bound=-1.0,
        upper_bound=1.0,
        familywise_alpha=config.familywise_alpha,
        hypotheses=len(config.hypothesis_ids),
        scheduled_looks=len(config.scheduled_looks),
    )
    positive = interval.lower > 0.0
    below_worthwhile = interval.upper < config.minimum_worthwhile_improvement
    if positive and below_worthwhile:
        return SequentialGateDecision(
            GateStatus.POSITIVE_BELOW_MWI,
            terminal=True,
            blocks=len(values),
            interval=interval,
            distribution_free_sensitivity=sensitivity,
            rule=(
                "effect is statistically positive but its entire interval is "
                "below the minimum worthwhile improvement"
            ),
            protocol_version=config.protocol_version,
            family_id=config.family_id,
            hypothesis_id=hypothesis_id,
        )
    if below_worthwhile:
        return SequentialGateDecision(
            GateStatus.FUTILITY,
            terminal=True,
            blocks=len(values),
            interval=interval,
            distribution_free_sensitivity=sensitivity,
            rule=(
                "upper simultaneous confidence bound is below the minimum worthwhile improvement"
            ),
            protocol_version=config.protocol_version,
            family_id=config.family_id,
            hypothesis_id=hypothesis_id,
        )
    if interval.lower > config.minimum_worthwhile_improvement:
        return SequentialGateDecision(
            GateStatus.EFFICACY,
            terminal=True,
            blocks=len(values),
            interval=interval,
            distribution_free_sensitivity=sensitivity,
            rule=("lower simultaneous confidence bound exceeds the minimum worthwhile improvement"),
            protocol_version=config.protocol_version,
            family_id=config.family_id,
            hypothesis_id=hypothesis_id,
        )
    if interval.half_width <= config.target_half_width:
        return SequentialGateDecision(
            GateStatus.PRECISE_INCONCLUSIVE,
            terminal=True,
            blocks=len(values),
            interval=interval,
            distribution_free_sensitivity=sensitivity,
            rule="precision target reached without efficacy or futility",
            protocol_version=config.protocol_version,
            family_id=config.family_id,
            hypothesis_id=hypothesis_id,
        )
    if len(values) == config.maximum_blocks:
        return SequentialGateDecision(
            GateStatus.MAXIMUM_REACHED,
            terminal=True,
            blocks=len(values),
            interval=interval,
            distribution_free_sensitivity=sensitivity,
            rule="predeclared maximum independent paired blocks reached",
            protocol_version=config.protocol_version,
            family_id=config.family_id,
            hypothesis_id=hypothesis_id,
        )
    return SequentialGateDecision(
        GateStatus.CONTINUE,
        terminal=False,
        blocks=len(values),
        interval=interval,
        distribution_free_sensitivity=sensitivity,
        rule="no terminal boundary crossed",
        protocol_version=config.protocol_version,
        family_id=config.family_id,
        hypothesis_id=hypothesis_id,
    )


def evaluate_sequential_family(
    improvements_by_hypothesis: Mapping[str, Sequence[float]],
    config: SequentialGateConfig | None = None,
    *,
    observed_family_id: str,
    observed_hypothesis_ids: Sequence[str],
) -> SequentialFamilyDecision:
    """Evaluate the synchronized conjunctive family and fail closed on drift.

    Confirmatory family success requires every registered endpoint to establish
    a worthwhile effect. Therefore one endpoint establishing practical
    futility terminates the global claim, whereas a favorable stop requires
    every endpoint to cross efficacy at the same accumulated look.
    """

    if config is None:
        config = SequentialGateConfig()
    config.validate_observed_family(
        family_id=observed_family_id,
        hypothesis_ids=observed_hypothesis_ids,
    )
    if set(improvements_by_hypothesis) != set(config.hypothesis_ids):
        raise ValueError("improvements mapping does not exactly cover the frozen family")
    lengths = {len(improvements_by_hypothesis[item]) for item in config.hypothesis_ids}
    if len(lengths) != 1:
        raise ValueError("every family endpoint must have the same completed block count")
    decisions = tuple(
        evaluate_sequential_gate(
            improvements_by_hypothesis[hypothesis_id],
            config,
            hypothesis_id=hypothesis_id,
            observed_family_id=observed_family_id,
            observed_hypothesis_ids=observed_hypothesis_ids,
        )
        for hypothesis_id in config.hypothesis_ids
    )
    blocks = decisions[0].blocks
    if decisions[0].status == GateStatus.NOT_A_LOOK:
        return SequentialFamilyDecision(
            status=GateStatus.NOT_A_LOOK,
            terminal=False,
            blocks=blocks,
            endpoint_decisions=decisions,
            rule="family decisions occur only at synchronized predeclared looks",
            protocol_version=config.protocol_version,
            family_id=config.family_id,
        )
    practically_futile = {
        GateStatus.FUTILITY,
        GateStatus.POSITIVE_BELOW_MWI,
    }
    if any(decision.status in practically_futile for decision in decisions):
        return SequentialFamilyDecision(
            status=GateStatus.FUTILITY,
            terminal=True,
            blocks=blocks,
            endpoint_decisions=decisions,
            rule=(
                "at least one endpoint rules out the minimum worthwhile effect, "
                "so the conjunctive family claim cannot succeed"
            ),
            protocol_version=config.protocol_version,
            family_id=config.family_id,
        )
    if all(decision.status == GateStatus.EFFICACY for decision in decisions):
        return SequentialFamilyDecision(
            status=GateStatus.EFFICACY,
            terminal=True,
            blocks=blocks,
            endpoint_decisions=decisions,
            rule="every registered endpoint establishes a minimum-worthwhile effect",
            protocol_version=config.protocol_version,
            family_id=config.family_id,
        )
    if blocks == config.maximum_blocks:
        return SequentialFamilyDecision(
            status=GateStatus.MAXIMUM_REACHED,
            terminal=True,
            blocks=blocks,
            endpoint_decisions=decisions,
            rule="hard maximum reached before a complete family decision",
            protocol_version=config.protocol_version,
            family_id=config.family_id,
        )
    if all(decision.terminal for decision in decisions):
        return SequentialFamilyDecision(
            status=GateStatus.PRECISE_INCONCLUSIVE,
            terminal=True,
            blocks=blocks,
            endpoint_decisions=decisions,
            rule="all endpoints are terminal but the conjunctive efficacy rule is not met",
            protocol_version=config.protocol_version,
            family_id=config.family_id,
        )
    return SequentialFamilyDecision(
        status=GateStatus.CONTINUE,
        terminal=False,
        blocks=blocks,
        endpoint_decisions=decisions,
        rule="no family-level terminal boundary crossed",
        protocol_version=config.protocol_version,
        family_id=config.family_id,
    )


@dataclass(frozen=True, slots=True)
class SequentialFeasibilityDiagnostics:
    """Pre-data width and effect thresholds for the frozen stopping design."""

    protocol_version: int
    family_id: str
    hypotheses: int
    looks: tuple[int, ...]
    familywise_alpha: float
    per_interval_alpha: float
    target_half_width: float
    assumed_standard_deviations: tuple[float, ...]
    student_t_critical_values: tuple[tuple[int, float], ...]
    maximum_sd_for_target_half_width: tuple[tuple[int, float], ...]
    worthwhile_efficacy_mean_thresholds_at_maximum: tuple[tuple[float, float], ...]
    original_endpointwise_hoeffding_radius_at_maximum: float
    familywise_hoeffding_radius_at_maximum: float
    conclusions: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))


def sequential_gate_feasibility(
    config: SequentialGateConfig | None = None,
    *,
    assumed_standard_deviations: Sequence[float] = (0.01, 0.03, 0.05, 0.10, 0.20),
) -> SequentialFeasibilityDiagnostics:
    """Quantify attainable widths before any accelerator observation."""

    if config is None:
        config = SequentialGateConfig()
    deviations = tuple(float(value) for value in assumed_standard_deviations)
    if not deviations or any(
        not math.isfinite(value) or value < 0.0 or value > 1.0 for value in deviations
    ):
        raise ValueError("assumed_standard_deviations must be finite values in [0, 1]")
    critical_values = tuple(
        (
            look,
            student_t_quantile(
                1.0 - config.per_interval_alpha / 2.0,
                look - 1,
            ),
        )
        for look in config.scheduled_looks
    )
    maximum_sd = tuple(
        (
            look,
            config.target_half_width * math.sqrt(look) / critical,
        )
        for look, critical in critical_values
    )
    maximum_look, maximum_critical = critical_values[-1]
    efficacy_thresholds = tuple(
        (
            deviation,
            config.minimum_worthwhile_improvement
            + maximum_critical * deviation / math.sqrt(maximum_look),
        )
        for deviation in deviations
    )
    original_alpha_at_maximum = 0.05 / (config.maximum_blocks * (config.maximum_blocks + 1))
    original_radius = 2.0 * math.sqrt(
        math.log(2.0 / original_alpha_at_maximum) / (2.0 * config.maximum_blocks)
    )
    familywise_radius = 2.0 * math.sqrt(
        math.log(2.0 / config.per_interval_alpha) / (2.0 * config.maximum_blocks)
    )
    return SequentialFeasibilityDiagnostics(
        protocol_version=config.protocol_version,
        family_id=config.family_id,
        hypotheses=len(config.hypothesis_ids),
        looks=config.scheduled_looks,
        familywise_alpha=config.familywise_alpha,
        per_interval_alpha=config.per_interval_alpha,
        target_half_width=config.target_half_width,
        assumed_standard_deviations=deviations,
        student_t_critical_values=critical_values,
        maximum_sd_for_target_half_width=maximum_sd,
        worthwhile_efficacy_mean_thresholds_at_maximum=efficacy_thresholds,
        original_endpointwise_hoeffding_radius_at_maximum=original_radius,
        familywise_hoeffding_radius_at_maximum=familywise_radius,
        conclusions=(
            "the version-1 Hoeffding rule cannot attain 0.03 half-width by 50 blocks",
            "the distribution-free familywise sensitivity interval also cannot attain it",
            "the primary Student-t rule can attain it only at the reported observed-SD thresholds",
            "early stopping is assumption-bounded and must be accompanied by sensitivity intervals",
        ),
    )


@dataclass(frozen=True, slots=True)
class SequentialCalibrationScenario:
    """One frozen Monte Carlo data-generating process."""

    name: str
    distribution: DiagnosticDistribution
    mean: float
    standard_deviation: float
    high_value_probability: float = 0.5

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("scenario name must not be empty")
        if not math.isfinite(self.mean) or not -1.0 <= self.mean <= 1.0:
            raise ValueError("scenario mean must be finite and in [-1, 1]")
        if (
            not math.isfinite(self.standard_deviation)
            or self.standard_deviation < 0.0
            or self.standard_deviation > 1.0
        ):
            raise ValueError("scenario standard_deviation must be in [0, 1]")
        if (
            not math.isfinite(self.high_value_probability)
            or not 0.0 < self.high_value_probability < 1.0
        ):
            raise ValueError("high_value_probability must be in (0, 1)")


DEFAULT_CALIBRATION_SCENARIOS: Final = (
    SequentialCalibrationScenario("null-normal", "normal", 0.0, 0.08),
    SequentialCalibrationScenario("small-normal", "normal", 0.01, 0.02),
    SequentialCalibrationScenario("worthwhile-normal", "normal", 0.06, 0.04),
    SequentialCalibrationScenario("worthwhile-low-variance", "normal", 0.06, 0.01),
    SequentialCalibrationScenario(
        "worthwhile-high-variance",
        "symmetric-two-point",
        0.06,
        0.30,
    ),
    SequentialCalibrationScenario(
        "adversarial-skewed-null",
        "skewed-bounded",
        0.0,
        math.sqrt(1.0 / 19.0),
        high_value_probability=0.05,
    ),
)


@dataclass(frozen=True, slots=True)
class ScenarioCalibrationResult:
    """Coverage and stopping frequencies for one Monte Carlo scenario."""

    name: str
    distribution: str
    mean: float
    standard_deviation: float
    trials: int
    primary_all_look_coverage_rate: float
    sensitivity_all_look_coverage_rate: float
    mean_blocks_used: float
    stop_rates: tuple[tuple[str, float], ...]

    def as_dict(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))


@dataclass(frozen=True, slots=True)
class SequentialCalibrationReport:
    """Deterministic pre-data calibration of operating characteristics."""

    protocol_version: int
    family_id: str
    trials: int
    seed_provenance: str
    scenarios: tuple[ScenarioCalibrationResult, ...]
    normal_null_familywise_noncoverage_rate: float
    normal_null_any_false_positive_rate: float
    familywise_target: float
    calibration_scope: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))


def _diagnostic_normal(
    rng: CounterRNG,
    *,
    trial: int,
    stream: str,
    draw: int,
) -> float:
    first = max(rng.uniform(trial, draw, f"{stream}/normal-u1", 0), 2.0**-53)
    second = rng.uniform(trial, draw, f"{stream}/normal-u2", 0)
    return math.sqrt(-2.0 * math.log(first)) * math.cos(2.0 * math.pi * second)


def _scenario_draw(
    scenario: SequentialCalibrationScenario,
    rng: CounterRNG,
    *,
    trial: int,
    draw: int,
    stream: str,
) -> float:
    if scenario.distribution == "normal":
        retry = 0
        while True:
            value = scenario.mean + scenario.standard_deviation * _diagnostic_normal(
                rng,
                trial=trial,
                stream=f"{stream}/retry-{retry}",
                draw=draw,
            )
            if -1.0 <= value <= 1.0:
                return value
            retry += 1
    probability = scenario.high_value_probability
    if scenario.distribution == "symmetric-two-point":
        return scenario.mean + (
            scenario.standard_deviation
            if rng.bernoulli(0.5, trial, draw, f"{stream}/symmetric")
            else -scenario.standard_deviation
        )
    if scenario.distribution == "skewed-bounded":
        high = scenario.mean + scenario.standard_deviation * math.sqrt(
            (1.0 - probability) / probability
        )
        low = scenario.mean - scenario.standard_deviation * math.sqrt(
            probability / (1.0 - probability)
        )
        if low < -1.0 - 1e-12 or high > 1.0 + 1e-12:
            raise ValueError("skewed bounded scenario support escapes [-1, 1]")
        return (
            min(1.0, high)
            if rng.bernoulli(probability, trial, draw, f"{stream}/skewed")
            else max(-1.0, low)
        )
    raise AssertionError(f"unknown diagnostic distribution {scenario.distribution!r}")


def sequential_gate_monte_carlo(
    config: SequentialGateConfig | None = None,
    *,
    trials: int = 2_000,
    seed: Seed = "fissionspec-sequential-calibration-v2",
    scenarios: Sequence[SequentialCalibrationScenario] = DEFAULT_CALIBRATION_SCENARIOS,
) -> SequentialCalibrationReport:
    """Calibrate coverage and stopping rates with counter-addressed draws.

    Normal scenarios audit the working Student-t model; the skewed bounded
    scenario intentionally exposes its non-robustness. The Hoeffding sensitivity
    interval is evaluated on every scenario. This simulation documents operating
    characteristics and does not replace the analytical Bonferroni guarantee.
    """

    if config is None:
        config = SequentialGateConfig()
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 100:
        raise ValueError("trials must be an integer of at least 100")
    scenario_values = tuple(scenarios)
    if not scenario_values or len({item.name for item in scenario_values}) != len(scenario_values):
        raise ValueError("scenarios must have unique non-empty names")
    rng = CounterRNG(seed)
    hypothesis_id = config.hypothesis_ids[0]
    results: list[ScenarioCalibrationResult] = []
    for scenario in scenario_values:
        primary_covered = 0
        sensitivity_covered = 0
        blocks_used = 0
        status_counts = {
            status.value: 0 for status in GateStatus if status != GateStatus.NOT_A_LOOK
        }
        for trial in range(trials):
            values = tuple(
                _scenario_draw(
                    scenario,
                    rng,
                    trial=trial,
                    draw=draw,
                    stream=f"scenario/{scenario.name}",
                )
                for draw in range(config.maximum_blocks)
            )
            primary_trial_covered = True
            sensitivity_trial_covered = True
            decision: SequentialGateDecision | None = None
            for look in config.scheduled_looks:
                current = evaluate_sequential_gate(
                    values[:look],
                    config,
                    hypothesis_id=hypothesis_id,
                    observed_family_id=config.family_id,
                    observed_hypothesis_ids=config.hypothesis_ids,
                )
                assert current.interval is not None
                assert current.distribution_free_sensitivity is not None
                primary_trial_covered = primary_trial_covered and (
                    current.interval.lower <= scenario.mean <= current.interval.upper
                )
                sensitivity_trial_covered = sensitivity_trial_covered and (
                    current.distribution_free_sensitivity.lower
                    <= scenario.mean
                    <= current.distribution_free_sensitivity.upper
                )
                if decision is None and current.terminal:
                    decision = current
            if decision is None:
                raise AssertionError("the maximum look must be terminal")
            primary_covered += int(primary_trial_covered)
            sensitivity_covered += int(sensitivity_trial_covered)
            blocks_used += decision.blocks
            status_counts[decision.status.value] += 1
        results.append(
            ScenarioCalibrationResult(
                name=scenario.name,
                distribution=scenario.distribution,
                mean=scenario.mean,
                standard_deviation=scenario.standard_deviation,
                trials=trials,
                primary_all_look_coverage_rate=primary_covered / trials,
                sensitivity_all_look_coverage_rate=sensitivity_covered / trials,
                mean_blocks_used=blocks_used / trials,
                stop_rates=tuple(
                    (status, count / trials) for status, count in sorted(status_counts.items())
                ),
            )
        )

    family_noncoverage = 0
    family_false_positive = 0
    null_sd = 0.08
    for trial in range(trials):
        any_noncoverage = False
        any_false_positive = False
        for hypothesis_index in range(len(config.hypothesis_ids)):
            values = tuple(
                null_sd
                * _diagnostic_normal(
                    rng,
                    trial=trial,
                    stream=f"family-null/{hypothesis_index}",
                    draw=draw,
                )
                for draw in range(config.maximum_blocks)
            )
            for look in config.scheduled_looks:
                interval = fixed_look_student_t_interval(
                    values[:look],
                    lower_bound=-1.0,
                    upper_bound=1.0,
                    familywise_alpha=config.familywise_alpha,
                    hypotheses=len(config.hypothesis_ids),
                    scheduled_looks=len(config.scheduled_looks),
                )
                any_noncoverage = any_noncoverage or not (interval.lower <= 0.0 <= interval.upper)
                any_false_positive = any_false_positive or interval.lower > 0.0
        family_noncoverage += int(any_noncoverage)
        family_false_positive += int(any_false_positive)
    return SequentialCalibrationReport(
        protocol_version=config.protocol_version,
        family_id=config.family_id,
        trials=trials,
        seed_provenance=rng.provenance,
        scenarios=tuple(results),
        normal_null_familywise_noncoverage_rate=family_noncoverage / trials,
        normal_null_any_false_positive_rate=family_false_positive / trials,
        familywise_target=config.familywise_alpha,
        calibration_scope=(
            "normal-null calibration uses all registered endpoints and scheduled looks",
            "scenario coverage means simultaneous coverage across every scheduled look",
            "normal draws outside [-1, 1] are rejected to respect the metric contract",
            "skewed bounded calibration is an adversarial sensitivity diagnostic",
            "Monte Carlo results are diagnostics, not proof or accelerator evidence",
        ),
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
    "DEFAULT_CALIBRATION_SCENARIOS",
    "PRIMARY_FAMILY_ID",
    "PRIMARY_HYPOTHESIS_IDS",
    "PRIMARY_METRICS",
    "PRIMARY_MODEL_PAIRS",
    "PRIMARY_VALIDATION_ANCHORS",
    "SEQUENTIAL_PROTOCOL_VERSION",
    "CalibrationRefinement",
    "DesignCell",
    "DiagnosticDistribution",
    "ExperimentSpendCaps",
    "GateStatus",
    "MetricDirection",
    "ScenarioCalibrationResult",
    "SequentialCalibrationReport",
    "SequentialCalibrationScenario",
    "SequentialFamilyDecision",
    "SequentialFeasibilityDiagnostics",
    "SequentialGateConfig",
    "SequentialGateDecision",
    "calibration_refinement_plan",
    "evaluate_sequential_family",
    "evaluate_sequential_gate",
    "paired_block_order",
    "select_farthest_cells",
    "sequential_gate_feasibility",
    "sequential_gate_monte_carlo",
    "symmetric_improvement",
]
