#!/usr/bin/env python3
"""Run deterministic, matched-seed synthetic FissionSpec experiments.

This is a mechanism-study harness, not a hardware benchmark. It independently
varies outcome-cache availability and token acceptance, evaluates all five
built-in policies against a synthetic latency surface, and writes paired
per-seed rows plus aggregate comparisons.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Final

from fissionspec.metrics import SimulationMetrics, summarize
from fissionspec.policies import policy_from_name
from fissionspec.profiles import HardwareProfile
from fissionspec.rng import CounterRNG
from fissionspec.simulator import simulate
from fissionspec.workload import RequestConfig, Workload

EVIDENCE_CLASS: Final = "synthetic-model"
WARNING: Final = "SYNTHETIC MODEL OUTPUT — NOT GPU MEASUREMENTS."
POLICIES: Final = (
    "saguaro-barrier",
    "spectre-parallel-padded",
    "immediate-fission",
    "fixed-coalesce",
    "fissionspec-horizon-2",
)
WORKLOAD_KINDS: Final = ("synchronized-cohort", "poisson", "bursty")
IMPLEMENTATION_FILES: Final = (
    "experiments/run_synthetic_sweep.py",
    "src/fissionspec/metrics.py",
    "src/fissionspec/model.py",
    "src/fissionspec/policies.py",
    "src/fissionspec/profiles.py",
    "src/fissionspec/rng.py",
    "src/fissionspec/simulator.py",
    "src/fissionspec/workload.py",
)


def synthetic_profile() -> HardwareProfile:
    """Return the immutable model surface used by the checked-in artifact."""

    return HardwareProfile(name="synthetic-reference-not-gpu")


@dataclass(frozen=True, slots=True)
class MechanismRegime:
    """One point in the cache-hit by token-acceptance factorial."""

    name: str
    cache_hit_probability: float
    token_acceptance_probability: float


@dataclass(frozen=True, slots=True)
class SweepConfig:
    """Complete deterministic input to one experiment sweep."""

    seeds: tuple[int, ...]
    request_count: int
    output_tokens: int
    speculation_length: int
    cache_hit_probabilities: tuple[float, ...]
    token_acceptance_probabilities: tuple[float, ...]
    tbt_slo_ms: float
    max_batch_size: int
    poisson_mean_ms: float
    burst_size: int
    burst_gap_ms: float
    burst_width_ms: float
    coalesce_ms: float
    max_wait_ms: float

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in self.seeds):
            raise ValueError("seeds must be integers")
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("seeds must be unique")
        for field_name in (
            "request_count",
            "output_tokens",
            "speculation_length",
            "max_batch_size",
            "burst_size",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        for field_name in (
            "cache_hit_probabilities",
            "token_acceptance_probabilities",
        ):
            probabilities = getattr(self, field_name)
            if len(probabilities) < 2 or len(probabilities) != len(set(probabilities)):
                raise ValueError(f"{field_name} must contain at least two distinct values")
            if any(
                isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not math.isfinite(probability)
                or not 0.0 <= probability <= 1.0
                for probability in probabilities
            ):
                raise ValueError(f"{field_name} values must be finite and in [0, 1]")
        for field_name in (
            "tbt_slo_ms",
            "poisson_mean_ms",
            "burst_gap_ms",
            "burst_width_ms",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{field_name} must be finite and positive")
        for field_name in ("coalesce_ms", "max_wait_ms"):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"{field_name} must be finite and non-negative")

    @staticmethod
    def _probability_label(value: float) -> str:
        return f"{value:.3f}".replace(".", "p")

    def regimes(self) -> tuple[MechanismRegime, ...]:
        """Return the complete ordered Cartesian product of both mechanisms."""

        return tuple(
            MechanismRegime(
                name=(
                    f"c{cache_index}-{self._probability_label(cache_probability)}"
                    f"_t{token_index}-{self._probability_label(token_probability)}"
                ),
                cache_hit_probability=cache_probability,
                token_acceptance_probability=token_probability,
            )
            for cache_index, cache_probability in enumerate(self.cache_hit_probabilities)
            for token_index, token_probability in enumerate(self.token_acceptance_probabilities)
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "seeds": list(self.seeds),
            "request_count": self.request_count,
            "output_tokens": self.output_tokens,
            "speculation_length": self.speculation_length,
            "cache_hit_probabilities": list(self.cache_hit_probabilities),
            "token_acceptance_probabilities": list(self.token_acceptance_probabilities),
            "tbt_slo_ms": self.tbt_slo_ms,
            "max_batch_size": self.max_batch_size,
            "poisson_mean_ms": self.poisson_mean_ms,
            "burst_size": self.burst_size,
            "burst_gap_ms": self.burst_gap_ms,
            "burst_width_ms": self.burst_width_ms,
            "coalesce_ms": self.coalesce_ms,
            "max_wait_ms": self.max_wait_ms,
        }


@dataclass(frozen=True, slots=True)
class ExperimentRow:
    """One policy result for one workload, regime, and common seed."""

    workload: str
    regime: str
    seed: int
    policy: str
    configured_cache_hit_probability: float
    configured_token_acceptance_probability: float
    observed_cache_hits: int
    observed_cache_misses: int
    observed_cache_hit_rate: float
    accepted_draft_tokens: int
    verifier_rounds: int
    mean_verifier_tokens_per_round: float
    requests: int
    output_tokens: int
    makespan_ms: float
    p50_tbt_ms: float
    p95_tbt_ms: float
    p99_tbt_ms: float
    throughput_tokens_per_s: float
    token_gap_slo_attainment: float
    request_tbt_slo_attainment: float
    tbt_request_goodput_tokens_per_s: float
    padded_verifier_slots: int
    direct_hit_delay_ms: float
    total_direct_hit_delay_ms: float
    target_launches: int
    draft_launches: int
    mean_batch: float

    @classmethod
    def from_metrics(
        cls,
        workload: str,
        regime: MechanismRegime,
        seed: int,
        metrics: SimulationMetrics,
    ) -> ExperimentRow:
        return cls(
            workload=workload,
            regime=regime.name,
            seed=seed,
            policy=metrics.policy_name,
            configured_cache_hit_probability=regime.cache_hit_probability,
            configured_token_acceptance_probability=(regime.token_acceptance_probability),
            observed_cache_hits=metrics.cache_hits,
            observed_cache_misses=metrics.cache_misses,
            observed_cache_hit_rate=metrics.observed_cache_hit_rate,
            accepted_draft_tokens=metrics.accepted_draft_tokens,
            verifier_rounds=metrics.verifier_rounds,
            mean_verifier_tokens_per_round=metrics.mean_verifier_tokens_per_round,
            requests=metrics.requests,
            output_tokens=metrics.output_tokens,
            makespan_ms=metrics.makespan_ms,
            p50_tbt_ms=metrics.p50_tbt_ms,
            p95_tbt_ms=metrics.p95_tbt_ms,
            p99_tbt_ms=metrics.p99_tbt_ms,
            throughput_tokens_per_s=metrics.throughput_tokens_per_s,
            token_gap_slo_attainment=metrics.token_gap_slo_attainment,
            request_tbt_slo_attainment=metrics.request_tbt_slo_attainment,
            tbt_request_goodput_tokens_per_s=(metrics.tbt_request_goodput_tokens_per_s),
            padded_verifier_slots=metrics.padded_verifier_slots,
            direct_hit_delay_ms=metrics.direct_hit_delay_ms,
            total_direct_hit_delay_ms=metrics.total_direct_hit_delay_ms,
            target_launches=metrics.target_launches,
            draft_launches=metrics.draft_launches,
            mean_batch=metrics.mean_batch,
        )

    def as_dict(self) -> dict[str, str | int | float]:
        return {
            "workload": self.workload,
            "regime": self.regime,
            "seed": self.seed,
            "policy": self.policy,
            "configured_cache_hit_probability": (self.configured_cache_hit_probability),
            "configured_token_acceptance_probability": (
                self.configured_token_acceptance_probability
            ),
            "observed_cache_hits": self.observed_cache_hits,
            "observed_cache_misses": self.observed_cache_misses,
            "observed_cache_hit_rate": self.observed_cache_hit_rate,
            "accepted_draft_tokens": self.accepted_draft_tokens,
            "verifier_rounds": self.verifier_rounds,
            "mean_verifier_tokens_per_round": self.mean_verifier_tokens_per_round,
            "requests": self.requests,
            "output_tokens": self.output_tokens,
            "makespan_ms": self.makespan_ms,
            "p50_tbt_ms": self.p50_tbt_ms,
            "p95_tbt_ms": self.p95_tbt_ms,
            "p99_tbt_ms": self.p99_tbt_ms,
            "throughput_tokens_per_s": self.throughput_tokens_per_s,
            "token_gap_slo_attainment": self.token_gap_slo_attainment,
            "request_tbt_slo_attainment": self.request_tbt_slo_attainment,
            "tbt_request_goodput_tokens_per_s": self.tbt_request_goodput_tokens_per_s,
            "padded_verifier_slots": self.padded_verifier_slots,
            "direct_hit_delay_ms": self.direct_hit_delay_ms,
            "total_direct_hit_delay_ms": self.total_direct_hit_delay_ms,
            "target_launches": self.target_launches,
            "draft_launches": self.draft_launches,
            "mean_batch": self.mean_batch,
        }


@dataclass(frozen=True, slots=True)
class AggregateRow:
    """Mean metrics and paired barrier deltas for one factorial cell."""

    workload: str
    regime: str
    policy: str
    configured_cache_hit_probability: float
    configured_token_acceptance_probability: float
    replicates: int
    observed_cache_hit_rate: float
    accepted_draft_tokens: float
    mean_verifier_tokens_per_round: float
    throughput_tokens_per_s: float
    p95_tbt_ms: float
    p99_tbt_ms: float
    token_gap_slo_attainment: float
    request_tbt_slo_attainment: float
    tbt_request_goodput_tokens_per_s: float
    padded_verifier_slots: float
    direct_hit_delay_ms: float
    target_launches: float
    mean_batch: float
    throughput_ratio_vs_barrier: float
    p95_tbt_delta_vs_barrier_ms: float

    def as_dict(self) -> dict[str, str | int | float]:
        return {
            "workload": self.workload,
            "regime": self.regime,
            "policy": self.policy,
            "configured_cache_hit_probability": (self.configured_cache_hit_probability),
            "configured_token_acceptance_probability": (
                self.configured_token_acceptance_probability
            ),
            "replicates": self.replicates,
            "observed_cache_hit_rate": self.observed_cache_hit_rate,
            "accepted_draft_tokens": self.accepted_draft_tokens,
            "mean_verifier_tokens_per_round": self.mean_verifier_tokens_per_round,
            "throughput_tokens_per_s": self.throughput_tokens_per_s,
            "p95_tbt_ms": self.p95_tbt_ms,
            "p99_tbt_ms": self.p99_tbt_ms,
            "token_gap_slo_attainment": self.token_gap_slo_attainment,
            "request_tbt_slo_attainment": self.request_tbt_slo_attainment,
            "tbt_request_goodput_tokens_per_s": self.tbt_request_goodput_tokens_per_s,
            "padded_verifier_slots": self.padded_verifier_slots,
            "direct_hit_delay_ms": self.direct_hit_delay_ms,
            "target_launches": self.target_launches,
            "mean_batch": self.mean_batch,
            "throughput_ratio_vs_barrier": self.throughput_ratio_vs_barrier,
            "p95_tbt_delta_vs_barrier_ms": self.p95_tbt_delta_vs_barrier_ms,
        }


def _request(
    config: SweepConfig,
    regime: MechanismRegime,
    request_id: str,
    arrival_ms: float,
) -> RequestConfig:
    return RequestConfig(
        request_id=request_id,
        arrival_ms=arrival_ms,
        output_tokens=config.output_tokens,
        speculation_length=config.speculation_length,
        cache_hit_probability=regime.cache_hit_probability,
        token_acceptance_probability=regime.token_acceptance_probability,
        tbt_slo_ms=config.tbt_slo_ms,
    )


def synchronized_workload(config: SweepConfig, regime: MechanismRegime, seed: int) -> Workload:
    """Construct one all-at-once cohort; the seed names the matched trace."""

    requests = tuple(
        _request(config, regime, f"sync-{index:04d}", 0.0) for index in range(config.request_count)
    )
    return Workload(requests, name=f"synchronized-cohort-{regime.name}-seed-{seed}")


def poisson_workload(config: SweepConfig, regime: MechanismRegime, seed: int) -> Workload:
    """Construct exponential inter-arrivals using counter-addressed draws."""

    generator = CounterRNG(seed)
    arrival_ms = 0.0
    requests: list[RequestConfig] = []
    for index in range(config.request_count):
        if index:
            uniform = generator.uniform("poisson-arrival", index, "workload-generation")
            arrival_ms += -config.poisson_mean_ms * math.log1p(-uniform)
        requests.append(_request(config, regime, f"poisson-{index:04d}", arrival_ms))
    return Workload(tuple(requests), name=f"poisson-{regime.name}-seed-{seed}")


def bursty_workload(config: SweepConfig, regime: MechanismRegime, seed: int) -> Workload:
    """Construct compact arrival bursts separated by deterministic idle gaps."""

    generator = CounterRNG(seed)
    requests: list[RequestConfig] = []
    for index in range(config.request_count):
        burst_index = index // config.burst_size
        jitter = (
            generator.uniform("bursty-arrival", index, "workload-generation")
            * config.burst_width_ms
        )
        arrival_ms = burst_index * config.burst_gap_ms + jitter
        requests.append(_request(config, regime, f"bursty-{index:04d}", arrival_ms))
    requests.sort(key=lambda request: (request.arrival_ms, request.request_id))
    return Workload(tuple(requests), name=f"bursty-{regime.name}-seed-{seed}")


def make_workload(kind: str, config: SweepConfig, regime: MechanismRegime, seed: int) -> Workload:
    """Dispatch to a stable workload generator."""

    if kind == "synchronized-cohort":
        return synchronized_workload(config, regime, seed)
    if kind == "poisson":
        return poisson_workload(config, regime, seed)
    if kind == "bursty":
        return bursty_workload(config, regime, seed)
    raise ValueError(f"unknown workload kind: {kind!r}")


def outcome_key_digests(workload: Workload, seed: int) -> tuple[str, str]:
    """Fingerprint both complete potential outcome-key spaces.

    A request can consume at most ``output_tokens`` speculative rounds because
    every verification emits at least one token. Hashing that superset records
    the common random-number streams shared by all scheduling policies.
    """

    rng = CounterRNG(seed)
    cache_digest = hashlib.sha256()
    token_digest = hashlib.sha256()
    for request in sorted(workload.requests, key=lambda item: item.request_id):
        encoded_id = request.request_id.encode("utf-8") + b"\x00"
        for round_id in range(request.output_tokens):
            encoded_round = round_id.to_bytes(8, "big")
            cache_digest.update(encoded_id)
            cache_digest.update(encoded_round)
            cache_digest.update(
                rng.uint64(request.request_id, round_id, "cache-hit").to_bytes(8, "big")
            )
            for draw in range(request.speculation_length - 1):
                token_digest.update(encoded_id)
                token_digest.update(encoded_round)
                token_digest.update(draw.to_bytes(8, "big"))
                token_digest.update(
                    rng.uint64(
                        request.request_id,
                        round_id,
                        "token-acceptance",
                        draw,
                    ).to_bytes(8, "big")
                )
    return cache_digest.hexdigest(), token_digest.hexdigest()


def implementation_sha256() -> str:
    """Fingerprint every source file that defines the checked-in model."""

    repository = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for relative_path in IMPLEMENTATION_FILES:
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\x00")
        digest.update((repository / relative_path).read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest()


def run_sweep(
    config: SweepConfig,
) -> tuple[list[ExperimentRow], list[dict[str, str | int]]]:
    """Run the full factorial with matched policy-level random streams."""

    profile = synthetic_profile()
    rows: list[ExperimentRow] = []
    fingerprints: list[dict[str, str | int]] = []
    regimes = config.regimes()
    for workload_kind in WORKLOAD_KINDS:
        for seed in config.seeds:
            fingerprint_workload = make_workload(workload_kind, config, regimes[0], seed)
            cache_digest, token_digest = outcome_key_digests(fingerprint_workload, seed)
            fingerprints.append(
                {
                    "workload": workload_kind,
                    "seed": seed,
                    "cache_hit_sha256": cache_digest,
                    "token_acceptance_sha256": token_digest,
                }
            )
            expected_tokens = config.request_count * config.output_tokens
            for regime in regimes:
                workload = make_workload(workload_kind, config, regime, seed)
                rng = CounterRNG(seed)
                for policy_name in POLICIES:
                    policy = policy_from_name(
                        policy_name,
                        coalesce_ms=config.coalesce_ms,
                        max_wait_ms=config.max_wait_ms,
                    )
                    result = simulate(
                        workload,
                        profile,
                        policy,
                        rng,
                        max_batch_size=config.max_batch_size,
                    )
                    metrics = summarize(result)
                    if (
                        metrics.requests != config.request_count
                        or metrics.output_tokens != expected_tokens
                    ):
                        raise RuntimeError(
                            f"{workload_kind}/{regime.name}/{seed}/{policy_name} "
                            "produced an incomplete trace"
                        )
                    rows.append(ExperimentRow.from_metrics(workload_kind, regime, seed, metrics))
    return rows, fingerprints


def aggregate_rows(rows: list[ExperimentRow], config: SweepConfig) -> list[AggregateRow]:
    """Average replicates while preserving paired comparisons to the barrier."""

    barrier_by_key = {
        (row.workload, row.regime, row.seed): row for row in rows if row.policy == "saguaro-barrier"
    }
    aggregates: list[AggregateRow] = []
    for workload in WORKLOAD_KINDS:
        for regime in config.regimes():
            for policy in POLICIES:
                selected = [
                    row
                    for row in rows
                    if row.workload == workload
                    and row.regime == regime.name
                    and row.policy == policy
                ]
                if not selected:
                    raise RuntimeError(f"no rows for {workload}/{regime.name}/{policy}")
                barrier_rows = [
                    barrier_by_key[(row.workload, row.regime, row.seed)] for row in selected
                ]
                aggregates.append(
                    AggregateRow(
                        workload=workload,
                        regime=regime.name,
                        policy=policy,
                        configured_cache_hit_probability=(regime.cache_hit_probability),
                        configured_token_acceptance_probability=(
                            regime.token_acceptance_probability
                        ),
                        replicates=len(selected),
                        observed_cache_hit_rate=fmean(
                            row.observed_cache_hit_rate for row in selected
                        ),
                        accepted_draft_tokens=fmean(row.accepted_draft_tokens for row in selected),
                        mean_verifier_tokens_per_round=fmean(
                            row.mean_verifier_tokens_per_round for row in selected
                        ),
                        throughput_tokens_per_s=fmean(
                            row.throughput_tokens_per_s for row in selected
                        ),
                        p95_tbt_ms=fmean(row.p95_tbt_ms for row in selected),
                        p99_tbt_ms=fmean(row.p99_tbt_ms for row in selected),
                        token_gap_slo_attainment=fmean(
                            row.token_gap_slo_attainment for row in selected
                        ),
                        request_tbt_slo_attainment=fmean(
                            row.request_tbt_slo_attainment for row in selected
                        ),
                        tbt_request_goodput_tokens_per_s=fmean(
                            row.tbt_request_goodput_tokens_per_s for row in selected
                        ),
                        padded_verifier_slots=fmean(row.padded_verifier_slots for row in selected),
                        direct_hit_delay_ms=fmean(row.direct_hit_delay_ms for row in selected),
                        target_launches=fmean(row.target_launches for row in selected),
                        mean_batch=fmean(row.mean_batch for row in selected),
                        throughput_ratio_vs_barrier=fmean(
                            row.throughput_tokens_per_s / barrier.throughput_tokens_per_s
                            for row, barrier in zip(selected, barrier_rows, strict=True)
                        ),
                        p95_tbt_delta_vs_barrier_ms=fmean(
                            row.p95_tbt_ms - barrier.p95_tbt_ms
                            for row, barrier in zip(selected, barrier_rows, strict=True)
                        ),
                    )
                )
    return aggregates


def _write_json(
    path: Path,
    config: SweepConfig,
    rows: list[ExperimentRow],
    aggregates: list[AggregateRow],
    fingerprints: list[dict[str, str | int]],
    csv_sha256: str,
) -> None:
    profile = synthetic_profile()
    document: dict[str, object] = {
        "schema_version": 4,
        "evidence_class": EVIDENCE_CLASS,
        "measurement_warning": WARNING,
        "profile": {
            "name": profile.name,
            "kind": "synthetic latency model",
            "gpu_measurement": False,
            "target_curve": [list(point) for point in profile.target_curve.points],
            "draft_curve": [list(point) for point in profile.draft_curve.points],
            "recovery_curve": [list(point) for point in profile.recovery_curve.points],
            "verifier_slot_ms": profile.verifier_slot_ms,
        },
        "method": {
            "generator": "experiments/run_synthetic_sweep.py",
            "implementation_files": list(IMPLEMENTATION_FILES),
            "implementation_sha256": implementation_sha256(),
            "determinism": "no wall-clock or hardware inputs",
            "comparison": "matched-seed common random numbers",
            "factorial": ("cache-hit probability x token-acceptance probability"),
            "cache_hit_key": ("(seed, request_id, round_id, 'cache-hit', draw=0)"),
            "token_acceptance_key": (
                "(seed, request_id, round_id, 'token-acceptance', draw=0..k-2)"
            ),
            "policies": list(POLICIES),
            "workloads": list(WORKLOAD_KINDS),
        },
        "config": config.as_dict(),
        "outcome_key_fingerprints": fingerprints,
        "per_seed_rows": {
            "format": "CSV",
            "path": "synthetic_sweep.csv",
            "row_count": len(rows),
            "sha256": csv_sha256,
        },
        "aggregates": [aggregate.as_dict() for aggregate in aggregates],
    }
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[ExperimentRow]) -> None:
    row_fields = tuple(rows[0].as_dict())
    fieldnames = ("evidence_class", "measurement_warning", *row_fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            payload: dict[str, str | int | float] = {
                "evidence_class": EVIDENCE_CLASS,
                "measurement_warning": WARNING,
            }
            payload.update(row.as_dict())
            writer.writerow(payload)


def write_results(
    output_dir: Path,
    config: SweepConfig,
    rows: list[ExperimentRow],
    fingerprints: list[dict[str, str | int]],
) -> tuple[Path, Path]:
    """Write deterministic JSON and CSV artifacts with explicit provenance."""

    output_dir.mkdir(parents=True, exist_ok=True)
    aggregates = aggregate_rows(rows, config)
    json_path = output_dir / "synthetic_sweep.json"
    csv_path = output_dir / "synthetic_sweep.csv"
    _write_csv(csv_path, rows)
    csv_sha256 = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    _write_json(
        json_path,
        config,
        rows,
        aggregates,
        fingerprints,
        csv_sha256,
    )
    return json_path, csv_path


def _parse_ints(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be comma-separated integers") from exc
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return values


def _parse_probabilities(raw: str) -> tuple[float, ...]:
    try:
        values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("probabilities must be comma-separated numbers") from exc
    if len(values) < 2:
        raise argparse.ArgumentTypeError("sweeps require at least two probability levels")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results"),
        help="directory for synthetic_sweep.{json,csv}",
    )
    parser.add_argument("--seeds", type=_parse_ints, default=(7, 17, 29))
    parser.add_argument("--requests", type=int, default=48)
    parser.add_argument("--output-tokens", type=int, default=48)
    parser.add_argument("--speculation-length", type=int, default=4)
    parser.add_argument(
        "--cache-hit-probabilities",
        type=_parse_probabilities,
        default=(0.70, 0.95),
        help="comma-separated cache-outcome hit probabilities",
    )
    parser.add_argument(
        "--token-acceptance-probabilities",
        type=_parse_probabilities,
        default=(0.55, 0.90),
        help="comma-separated per-candidate token acceptance probabilities",
    )
    parser.add_argument("--tbt-slo-ms", type=float, default=10.0)
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--poisson-mean-ms", type=float, default=0.45)
    parser.add_argument("--burst-size", type=int, default=12)
    parser.add_argument("--burst-gap-ms", type=float, default=12.0)
    parser.add_argument("--burst-width-ms", type=float, default=0.35)
    parser.add_argument("--coalesce-ms", type=float, default=1.0)
    parser.add_argument("--max-wait-ms", type=float, default=2.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = SweepConfig(
        seeds=args.seeds,
        request_count=args.requests,
        output_tokens=args.output_tokens,
        speculation_length=args.speculation_length,
        cache_hit_probabilities=args.cache_hit_probabilities,
        token_acceptance_probabilities=args.token_acceptance_probabilities,
        tbt_slo_ms=args.tbt_slo_ms,
        max_batch_size=args.max_batch_size,
        poisson_mean_ms=args.poisson_mean_ms,
        burst_size=args.burst_size,
        burst_gap_ms=args.burst_gap_ms,
        burst_width_ms=args.burst_width_ms,
        coalesce_ms=args.coalesce_ms,
        max_wait_ms=args.max_wait_ms,
    )
    rows, fingerprints = run_sweep(config)
    json_path, csv_path = write_results(args.output_dir, config, rows, fingerprints)
    print(WARNING)
    print(f"wrote {len(rows)} matched policy rows to {json_path} and {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
