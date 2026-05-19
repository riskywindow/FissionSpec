"""Dependency-free statistical primitives for paired serving experiments.

The functions in this module deliberately make the experimental unit explicit.
Policy comparisons should be paired within a seed/trace and resampled at the
independent seed/trace cluster, never by treating requests from one run as
independent replications.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Literal, cast

from .rng import CounterRNG, Seed

Direction = Literal["higher", "lower"]


def _finite_values(values: Sequence[float], *, field: str, minimum: int) -> tuple[float, ...]:
    converted = tuple(float(value) for value in values)
    if len(converted) < minimum:
        raise ValueError(f"{field} needs at least {minimum} observations")
    if any(not math.isfinite(value) for value in converted):
        raise ValueError(f"{field} must contain only finite observations")
    return converted


def _probability(value: float, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 < value < 1.0
    ):
        raise ValueError(f"{field} must be finite and strictly between zero and one")
    return float(value)


def _positive_integer(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _sample_standard_deviation(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise ValueError("sample standard deviation needs at least two observations")
    center = _mean(values)
    return math.sqrt(math.fsum((value - center) ** 2 for value in values) / (len(values) - 1))


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


@dataclass(frozen=True, slots=True)
class PairedEffect:
    """Descriptive paired effect, oriented so positive means improvement."""

    observations: int
    direction: Direction
    baseline_mean: float
    candidate_mean: float
    raw_mean_difference: float
    oriented_mean_improvement: float
    relative_mean_difference: float | None
    paired_standardized_improvement: float | None
    paired_difference_standard_deviation: float
    probability_of_improvement: float

    def as_dict(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))


def paired_effect_size(
    candidate: Sequence[float],
    baseline: Sequence[float],
    *,
    direction: Direction,
) -> PairedEffect:
    """Return paired mean, relative, standardized, and win-probability effects.

    ``paired_standardized_improvement`` is Cohen's ``d_z``: oriented mean
    paired difference divided by the sample standard deviation of paired
    differences. It is undefined when every paired difference is identical.
    """

    if direction not in {"higher", "lower"}:
        raise ValueError("direction must be 'higher' or 'lower'")
    candidate_values = _finite_values(candidate, field="candidate", minimum=2)
    baseline_values = _finite_values(baseline, field="baseline", minimum=2)
    if len(candidate_values) != len(baseline_values):
        raise ValueError("candidate and baseline must have equal paired lengths")
    differences = tuple(
        candidate_value - baseline_value
        for candidate_value, baseline_value in zip(
            candidate_values,
            baseline_values,
            strict=True,
        )
    )
    orientation = 1.0 if direction == "higher" else -1.0
    oriented = tuple(orientation * difference for difference in differences)
    baseline_mean = _mean(baseline_values)
    candidate_mean = _mean(candidate_values)
    difference_mean = _mean(differences)
    difference_sd = _sample_standard_deviation(differences)
    standardized = orientation * difference_mean / difference_sd if difference_sd > 0.0 else None
    wins = sum(value > 0.0 for value in oriented)
    ties = sum(value == 0.0 for value in oriented)
    return PairedEffect(
        observations=len(differences),
        direction=direction,
        baseline_mean=baseline_mean,
        candidate_mean=candidate_mean,
        raw_mean_difference=difference_mean,
        oriented_mean_improvement=orientation * difference_mean,
        relative_mean_difference=(
            difference_mean / abs(baseline_mean) if baseline_mean != 0.0 else None
        ),
        paired_standardized_improvement=standardized,
        paired_difference_standard_deviation=difference_sd,
        probability_of_improvement=(wins + 0.5 * ties) / len(oriented),
    )


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    """Percentile interval from deterministic paired cluster resampling."""

    method: str
    confidence_level: float
    point_estimate: float
    lower: float
    upper: float
    bootstrap_standard_error: float
    clusters: int
    resamples: int
    seed_provenance: str
    resample_fingerprint_sha256: str
    estimand: str

    def as_dict(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))


def _uniform_index(
    rng: CounterRNG,
    *,
    resample: int,
    draw: int,
    upper: int,
) -> int:
    """Draw an exactly uniform index with deterministic rejection sampling."""

    modulus = 1 << 64
    acceptance_limit = modulus - modulus % upper
    retry = 0
    while True:
        value = rng.uint64(
            "paired-cluster-bootstrap",
            resample,
            f"cluster-index/{draw}",
            retry,
        )
        if value < acceptance_limit:
            return value % upper
        retry += 1


def paired_cluster_bootstrap(
    differences_by_cluster: Mapping[str, Sequence[float]],
    *,
    confidence_level: float = 0.95,
    resamples: int = 10_000,
    seed: Seed = "fissionspec-paired-cluster-bootstrap-v1",
) -> BootstrapInterval:
    """Bootstrap the equally weighted mean of paired cluster means.

    Each mapping value may contain several matched observations from one
    independent cluster. The cluster is reduced to its mean before resampling,
    preventing pseudoreplication when one seed contributes more rows.
    """

    confidence = _probability(confidence_level, field="confidence_level")
    _positive_integer(resamples, field="resamples")
    if resamples < 100:
        raise ValueError("resamples must be at least 100 for an interval")
    if len(differences_by_cluster) < 2:
        raise ValueError("paired cluster bootstrap needs at least two clusters")
    cluster_ids = tuple(sorted(differences_by_cluster))
    if any(not cluster_id for cluster_id in cluster_ids):
        raise ValueError("cluster identifiers must be non-empty strings")
    cluster_means = tuple(
        _mean(
            _finite_values(
                differences_by_cluster[cluster_id],
                field=f"cluster {cluster_id!r}",
                minimum=1,
            )
        )
        for cluster_id in cluster_ids
    )
    rng = CounterRNG(seed)
    fingerprint = hashlib.sha256()
    estimates: list[float] = []
    for resample in range(resamples):
        selected: list[float] = []
        for draw in range(len(cluster_means)):
            index = _uniform_index(
                rng,
                resample=resample,
                draw=draw,
                upper=len(cluster_means),
            )
            fingerprint.update(index.to_bytes(8, "big"))
            selected.append(cluster_means[index])
        estimates.append(_mean(selected))
    tail = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        method="paired-percentile-cluster-bootstrap",
        confidence_level=confidence,
        point_estimate=_mean(cluster_means),
        lower=_quantile(estimates, tail),
        upper=_quantile(estimates, 1.0 - tail),
        bootstrap_standard_error=_sample_standard_deviation(estimates),
        clusters=len(cluster_means),
        resamples=resamples,
        seed_provenance=rng.provenance,
        resample_fingerprint_sha256=fingerprint.hexdigest(),
        estimand="equally weighted mean of within-cluster paired differences",
    )


@dataclass(frozen=True, slots=True)
class ConfidenceSequencePoint:
    """One simultaneous confidence interval for a bounded running mean."""

    observations: int
    mean: float
    lower: float
    upper: float
    half_width: float
    alpha_spent_at_look: float

    def as_dict(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))


def bounded_mean_confidence_sequence(
    values: Sequence[float],
    *,
    lower_bound: float,
    upper_bound: float,
    confidence_level: float = 0.95,
) -> tuple[ConfidenceSequencePoint, ...]:
    """Return an anytime-valid Hoeffding confidence sequence.

    At look ``t`` this uses ``alpha_t = alpha / (t(t+1))``. These terms sum to
    ``alpha``, so a union bound gives simultaneous coverage across every
    returned look. Bounds must be fixed before inspecting the observations.
    """

    observations = _finite_values(values, field="values", minimum=1)
    confidence = _probability(confidence_level, field="confidence_level")
    if (
        not math.isfinite(lower_bound)
        or not math.isfinite(upper_bound)
        or lower_bound > upper_bound
    ):
        raise ValueError("bounds must be finite and ordered")
    if any(value < lower_bound or value > upper_bound for value in observations):
        raise ValueError("every observation must lie inside the predeclared bounds")
    alpha = 1.0 - confidence
    value_range = upper_bound - lower_bound
    running_sum = 0.0
    sequence: list[ConfidenceSequencePoint] = []
    for index, value in enumerate(observations, start=1):
        running_sum += value
        center = running_sum / index
        alpha_at_look = alpha / (index * (index + 1))
        radius = (
            value_range * math.sqrt(math.log(2.0 / alpha_at_look) / (2.0 * index))
            if value_range
            else 0.0
        )
        lower = max(lower_bound, center - radius)
        upper = min(upper_bound, center + radius)
        sequence.append(
            ConfidenceSequencePoint(
                observations=index,
                mean=center,
                lower=lower,
                upper=upper,
                half_width=(upper - lower) / 2.0,
                alpha_spent_at_look=alpha_at_look,
            )
        )
    return tuple(sequence)


@dataclass(frozen=True, slots=True)
class PrecisionStoppingResult:
    """First anytime-valid look satisfying a predeclared precision target."""

    method: str
    confidence_level: float
    lower_bound: float
    upper_bound: float
    target_half_width: float
    minimum_observations: int
    available_observations: int
    observations_used: int
    reached_precision: bool
    final_interval: ConfidenceSequencePoint
    stopping_rule: str

    def as_dict(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))


def precision_stopping(
    values: Sequence[float],
    *,
    lower_bound: float,
    upper_bound: float,
    target_half_width: float,
    minimum_observations: int = 2,
    confidence_level: float = 0.95,
) -> PrecisionStoppingResult:
    """Apply a valid optional-stopping rule to bounded paired differences."""

    if (
        isinstance(target_half_width, bool)
        or not isinstance(target_half_width, (int, float))
        or not math.isfinite(target_half_width)
        or target_half_width <= 0.0
    ):
        raise ValueError("target_half_width must be finite and positive")
    minimum = _positive_integer(minimum_observations, field="minimum_observations")
    sequence = bounded_mean_confidence_sequence(
        values,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        confidence_level=confidence_level,
    )
    selected = sequence[-1]
    reached = False
    for point in sequence[min(minimum - 1, len(sequence)) :]:
        if point.half_width <= target_half_width:
            selected = point
            reached = True
            break
    return PrecisionStoppingResult(
        method="alpha-spending-hoeffding-confidence-sequence",
        confidence_level=float(confidence_level),
        lower_bound=float(lower_bound),
        upper_bound=float(upper_bound),
        target_half_width=float(target_half_width),
        minimum_observations=minimum,
        available_observations=len(sequence),
        observations_used=selected.observations,
        reached_precision=reached,
        final_interval=selected,
        stopping_rule=(
            "first look at or after minimum_observations with interval half-width <= target"
        ),
    )


@dataclass(frozen=True, slots=True)
class MultiplicityMetadata:
    """Predeclared Bonferroni family metadata for simultaneous intervals."""

    family_id: str
    method: str
    familywise_alpha: float
    hypotheses: tuple[str, ...]
    per_hypothesis_alpha: float
    simultaneous_per_hypothesis_confidence_level: float
    confirmatory: bool

    def as_dict(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))


def bonferroni_metadata(
    hypothesis_ids: Sequence[str],
    *,
    family_id: str,
    familywise_alpha: float = 0.05,
    confirmatory: bool = False,
) -> MultiplicityMetadata:
    """Declare one family and its conservative simultaneous CI level."""

    if not isinstance(family_id, str) or not family_id:
        raise ValueError("family_id must be a non-empty string")
    alpha = _probability(familywise_alpha, field="familywise_alpha")
    hypotheses = tuple(hypothesis_ids)
    if not hypotheses or any(not isinstance(item, str) or not item for item in hypotheses):
        raise ValueError("hypothesis identifiers must be non-empty strings")
    if len(hypotheses) != len(set(hypotheses)):
        raise ValueError("hypothesis identifiers must be unique")
    per_hypothesis = alpha / len(hypotheses)
    return MultiplicityMetadata(
        family_id=family_id,
        method="bonferroni-simultaneous-confidence-intervals",
        familywise_alpha=alpha,
        hypotheses=hypotheses,
        per_hypothesis_alpha=per_hypothesis,
        simultaneous_per_hypothesis_confidence_level=1.0 - per_hypothesis,
        confirmatory=confirmatory,
    )


def _standard_normal_quantile(probability: float) -> float:
    """Acklam's dependency-free inverse standard-normal approximation."""

    probability = _probability(probability, field="probability")
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    low = 0.02425
    high = 1.0 - low
    if probability < low:
        q = math.sqrt(-2.0 * math.log(probability))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if probability > high:
        q = math.sqrt(-2.0 * math.log1p(-probability))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    q = probability - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )


@dataclass(frozen=True, slots=True)
class ReplicationPlan:
    """Normal-approximation planning result for a paired mean comparison."""

    method: str
    current_replications: int
    recommended_replications: int
    additional_replications: int
    familywise_alpha: float
    hypotheses: int
    per_hypothesis_alpha: float
    target_power: float
    two_sided: bool
    minimum_detectable_standardized_effect: float
    pilot_paired_difference_standard_deviation: float
    implied_raw_minimum_detectable_effect: float
    assumptions: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))


def paired_replication_plan(
    pilot_differences: Sequence[float],
    *,
    minimum_detectable_standardized_effect: float,
    familywise_alpha: float = 0.05,
    target_power: float = 0.8,
    hypotheses: int = 1,
    two_sided: bool = True,
) -> ReplicationPlan:
    """Plan independent paired replications using a transparent normal approximation."""

    differences = _finite_values(
        pilot_differences,
        field="pilot_differences",
        minimum=2,
    )
    if (
        isinstance(minimum_detectable_standardized_effect, bool)
        or not isinstance(minimum_detectable_standardized_effect, (int, float))
        or not math.isfinite(minimum_detectable_standardized_effect)
        or minimum_detectable_standardized_effect <= 0.0
    ):
        raise ValueError("minimum detectable standardized effect must be finite and positive")
    alpha = _probability(familywise_alpha, field="familywise_alpha")
    power = _probability(target_power, field="target_power")
    if power <= 0.5:
        raise ValueError("target_power must exceed 0.5 for replication planning")
    hypothesis_count = _positive_integer(hypotheses, field="hypotheses")
    if not isinstance(two_sided, bool):
        raise TypeError("two_sided must be a bool")
    per_hypothesis_alpha = alpha / hypothesis_count
    critical_probability = 1.0 - per_hypothesis_alpha / (2.0 if two_sided else 1.0)
    critical_value = _standard_normal_quantile(critical_probability)
    power_value = _standard_normal_quantile(power)
    effect = float(minimum_detectable_standardized_effect)
    recommended = max(2, math.ceil(((critical_value + power_value) / effect) ** 2))
    pilot_sd = _sample_standard_deviation(differences)
    return ReplicationPlan(
        method="normal-approximation-paired-mean-with-bonferroni-alpha",
        current_replications=len(differences),
        recommended_replications=recommended,
        additional_replications=max(0, recommended - len(differences)),
        familywise_alpha=alpha,
        hypotheses=hypothesis_count,
        per_hypothesis_alpha=per_hypothesis_alpha,
        target_power=power,
        two_sided=two_sided,
        minimum_detectable_standardized_effect=effect,
        pilot_paired_difference_standard_deviation=pilot_sd,
        implied_raw_minimum_detectable_effect=effect * pilot_sd,
        assumptions=(
            "independent paired seed/trace replications",
            "approximately normal sampling distribution of the paired mean",
            "pilot paired-difference variance is representative",
            "planning calculation only; not a post-hoc significance claim",
        ),
    )
