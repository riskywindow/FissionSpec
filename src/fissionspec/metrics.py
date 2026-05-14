"""Trace metrics and closed-form batching externality calculations."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from .model import SimulationResult


def _probabilities(values: Iterable[float]) -> tuple[float, ...]:
    probabilities = tuple(float(value) for value in values)
    if not probabilities:
        raise ValueError("at least one probability is required")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
        raise ValueError("probabilities must be finite and in [0, 1]")
    return probabilities


def batch_fallback_probability(hit_probabilities: Iterable[float]) -> float:
    """Return ``P(any miss) = 1 - product(p_i)`` for a verifier batch."""

    probabilities = _probabilities(hit_probabilities)
    return 1.0 - math.prod(probabilities)


def expected_collateral_hit_stalls(hit_probabilities: Iterable[float]) -> float:
    """Expected hit rows stalled solely because another row misses.

    For row ``i`` this event has probability ``p_i * (1 - product(p_j,
    j != i))``.  Summing it exposes the externality hidden by aggregate batch
    fallback rate.
    """

    probabilities = _probabilities(hit_probabilities)
    total = 0.0
    for index, probability in enumerate(probabilities):
        other_hit_probability = math.prod(
            value for other_index, value in enumerate(probabilities) if other_index != index
        )
        total += probability * (1.0 - other_hit_probability)
    return total


def head_of_line_amplification(hit_probabilities: Iterable[float]) -> float:
    """Return barrier recovery work divided by decoupled recovery work.

    A barrier makes all ``n`` rows wait with probability ``P(any miss)``, so
    its expected stalled-row work is ``n * P(any miss)``.  Immediate fission
    stalls only misses, with expectation ``sum(1 - p_i)``.  Their ratio is the
    head-of-line amplification.  An all-hit batch has no stalled work and is
    assigned the neutral ratio 1.
    """

    probabilities = _probabilities(hit_probabilities)
    expected_misses = sum(1.0 - value for value in probabilities)
    if expected_misses == 0.0:
        return 1.0
    barrier_work = len(probabilities) * batch_fallback_probability(probabilities)
    return barrier_work / expected_misses


hol_amplification = head_of_line_amplification
probability_of_batch_fallback = batch_fallback_probability


def percentile(values: Iterable[float], quantile: float) -> float:
    """Linearly interpolated quantile using the ``(n - 1) * q`` convention."""

    if not math.isfinite(quantile) or not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be finite and in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


@dataclass(frozen=True, slots=True)
class SimulationMetrics:
    """Publication-oriented aggregate metrics for one completed trace.

    TBT excludes time-to-first-token and includes zero gaps inside an accepted
    speculative block.  ``slo_attainment`` is the fraction of emitted
    inter-token gaps at or below that request's TBT SLO.
    """

    policy_name: str
    requests: int
    output_tokens: int
    makespan_ms: float
    p50_tbt_ms: float
    p95_tbt_ms: float
    p99_tbt_ms: float
    throughput_tokens_per_s: float
    slo_attainment: float
    padded_verifier_slots: int
    hit_externality_ms: float
    total_hit_externality_ms: float
    target_launches: int
    draft_launches: int
    mean_batch: float

    @property
    def p50_tbt(self) -> float:
        return self.p50_tbt_ms

    @property
    def p95_tbt(self) -> float:
        return self.p95_tbt_ms

    @property
    def p99_tbt(self) -> float:
        return self.p99_tbt_ms

    @property
    def throughput(self) -> float:
        return self.throughput_tokens_per_s

    @property
    def launches(self) -> int:
        return self.target_launches

    def as_dict(self) -> dict[str, str | int | float]:
        """Return a JSON-friendly representation."""

        return asdict(self)


def summarize(result: SimulationResult) -> SimulationMetrics:
    """Compute all required aggregate metrics from an immutable trace."""

    tbt_values: list[float] = []
    slo_successes = 0
    total_hits = 0
    total_externality = 0.0
    for request in result.requests:
        gaps = request.inter_token_times_ms
        tbt_values.extend(gaps)
        slo_successes += sum(gap <= request.tbt_slo_ms for gap in gaps)
        total_hits += request.hits
        total_externality += request.hit_externality_ms

    makespan = result.makespan_ms
    throughput = (
        result.total_output_tokens * 1000.0 / makespan if makespan > 0.0 else 0.0
    )
    slo_attainment = slo_successes / len(tbt_values) if tbt_values else 1.0
    target_launch_count = len(result.target_launches)
    mean_batch = (
        sum(launch.effective_batch_size for launch in result.target_launches)
        / target_launch_count
        if target_launch_count
        else 0.0
    )
    return SimulationMetrics(
        policy_name=result.policy_name,
        requests=len(result.requests),
        output_tokens=result.total_output_tokens,
        makespan_ms=makespan,
        p50_tbt_ms=percentile(tbt_values, 0.50),
        p95_tbt_ms=percentile(tbt_values, 0.95),
        p99_tbt_ms=percentile(tbt_values, 0.99),
        throughput_tokens_per_s=throughput,
        slo_attainment=slo_attainment,
        padded_verifier_slots=result.padded_verifier_slots,
        hit_externality_ms=(total_externality / total_hits if total_hits else 0.0),
        total_hit_externality_ms=total_externality,
        target_launches=target_launch_count,
        draft_launches=len(result.draft_launches),
        mean_batch=mean_batch,
    )


compute_metrics = summarize


@dataclass(frozen=True, slots=True)
class CounterfactualMetrics:
    """Paired candidate-minus-baseline differences from common random draws."""

    baseline: SimulationMetrics
    candidate: SimulationMetrics
    p50_tbt_delta_ms: float
    p95_tbt_delta_ms: float
    p99_tbt_delta_ms: float
    throughput_delta_tokens_per_s: float
    throughput_ratio: float
    slo_attainment_delta: float
    padded_verifier_slots_delta: int
    hit_externality_delta_ms: float
    target_launches_delta: int
    mean_batch_delta: float


def counterfactual_metrics(
    candidate: SimulationResult, baseline: SimulationResult
) -> CounterfactualMetrics:
    """Compare paired policy traces produced from the same keyed RNG seed."""

    candidate_metrics = summarize(candidate)
    baseline_metrics = summarize(baseline)
    if candidate_metrics.output_tokens != baseline_metrics.output_tokens:
        raise ValueError("counterfactual traces must contain the same output tokens")
    baseline_throughput = baseline_metrics.throughput_tokens_per_s
    throughput_ratio = (
        candidate_metrics.throughput_tokens_per_s / baseline_throughput
        if baseline_throughput > 0.0
        else math.inf
    )
    return CounterfactualMetrics(
        baseline=baseline_metrics,
        candidate=candidate_metrics,
        p50_tbt_delta_ms=candidate_metrics.p50_tbt_ms - baseline_metrics.p50_tbt_ms,
        p95_tbt_delta_ms=candidate_metrics.p95_tbt_ms - baseline_metrics.p95_tbt_ms,
        p99_tbt_delta_ms=candidate_metrics.p99_tbt_ms - baseline_metrics.p99_tbt_ms,
        throughput_delta_tokens_per_s=(
            candidate_metrics.throughput_tokens_per_s
            - baseline_metrics.throughput_tokens_per_s
        ),
        throughput_ratio=throughput_ratio,
        slo_attainment_delta=(
            candidate_metrics.slo_attainment - baseline_metrics.slo_attainment
        ),
        padded_verifier_slots_delta=(
            candidate_metrics.padded_verifier_slots
            - baseline_metrics.padded_verifier_slots
        ),
        hit_externality_delta_ms=(
            candidate_metrics.hit_externality_ms
            - baseline_metrics.hit_externality_ms
        ),
        target_launches_delta=(
            candidate_metrics.target_launches - baseline_metrics.target_launches
        ),
        mean_batch_delta=candidate_metrics.mean_batch - baseline_metrics.mean_batch,
    )
