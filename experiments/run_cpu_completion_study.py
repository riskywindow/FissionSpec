#!/usr/bin/env python3
"""Reproduce the complete GPU-free FissionSpec evaluation bundle.

This experiment intentionally keeps three evidence strata separate:

* the decoder-policy simulator compares barrier, immediate fission, fixed
  coalescing, and the horizon-2 controller;
* the pre-realized scheduler harness compares FIFO, SPECTRE hybrid, EXSpec,
  and myopic slack without changing semantic outcomes; and
* the one-round fidelity harness exercises cache, context, transport, and
  remote-draft mechanisms.

No ranking is computed across those strata. Every generated artifact is
explicitly labeled as CPU simulation/model evidence, never a GPU measurement.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Final, Literal, TypeAlias, cast

from fissionspec.artifacts import (
    SIMULATION_WARNING,
    canonical_json_bytes,
    environment_manifest,
    implementation_sha256,
    sha256_document,
    simulation_trace_document,
)
from fissionspec.baselines import (
    BackgroundDraftJob,
    BaselineCostModel,
    BaselineResult,
    BaselineScheduler,
    DeterministicBaselineSimulator,
    EXSpecSlidingPoolScheduler,
    FIFOScheduler,
    MyopicSlackScheduler,
    PreRealizedTrace,
    RealizedRequest,
    RealizedStep,
    SpectreCalibration,
    SPECTREHybridScheduler,
    assert_semantic_equivalence,
)
from fissionspec.fidelity import (
    ContextCostModel,
    FidelityConfig,
    FidelityRequest,
    OutcomeClass,
    RemoteDraftConfig,
    simulate_fidelity_trace,
)
from fissionspec.general_oracle import (
    DispatchEvent,
    ExactLatencySurface,
    OracleCapacity,
    OracleJob,
    OracleProblem,
    OracleSearchLimits,
    OracleWaitConfig,
    ScheduleEvaluation,
    WaitEvent,
    WaitKind,
    objective_gap,
    score_completion_times,
    solve_general_oracle,
    verify_general_oracle_certificate,
    work_conserving_edf,
)
from fissionspec.metrics import percentile, summarize
from fissionspec.model import SimulationResult
from fissionspec.policies import (
    DispatchContext,
    FissionSpecPolicy,
    FixedCoalescePolicy,
    ImmediateFissionPolicy,
    SaguaroBarrierPolicy,
    SchedulingPolicy,
)
from fissionspec.profiles import HardwareProfile, LatencyCurve
from fissionspec.rng import CounterRNG
from fissionspec.simulator import simulate
from fissionspec.statistics import (
    bonferroni_metadata,
    paired_cluster_bootstrap,
    paired_effect_size,
)
from fissionspec.workload import RequestConfig, Workload
from fissionspec.workload_generators import (
    ArrivalTrace,
    load_trace_csv,
    mmpp_arrivals,
    pareto_arrivals,
    poisson_arrivals,
    workload_from_arrivals,
)

SCHEMA_VERSION: Final = 1
WARNING: Final = SIMULATION_WARNING
CLAIM_BOUNDARY: Final = (
    "All values are deterministic CPU simulation/model outputs. They do not "
    "measure GPU kernels, CUDA graphs, accelerator memory, power, or production throughput."
)
StudyMode = Literal["ci", "full"]
MetricRow: TypeAlias = dict[str, str | int | float]
JsonObject: TypeAlias = dict[str, object]

WORKLOAD_KINDS: Final = (
    "synchronized",
    "poisson",
    "mmpp-exact",
    "pareto-heavy-tail",
    "heterogeneous",
    "trace-replay",
)
MAIN_REFERENCE: Final = "saguaro-barrier"
SCHEDULER_REFERENCE: Final = "fifo-ordinary-reference"
HEADLINE_POLICY: Final = "fissionspec-horizon-2"
MAIN_HARNESS: Final = "decoder-policy-simulator"
SCHEDULER_HARNESS: Final = "pre-realized-scheduler-abstraction"
FIDELITY_HARNESS: Final = "one-round-fidelity-model"

IMPLEMENTATION_PATHS: Final = (
    "experiments/run_cpu_completion_study.py",
    "src/fissionspec/artifacts.py",
    "src/fissionspec/baselines.py",
    "src/fissionspec/fidelity.py",
    "src/fissionspec/general_oracle.py",
    "src/fissionspec/metrics.py",
    "src/fissionspec/policies.py",
    "src/fissionspec/profiles.py",
    "src/fissionspec/rng.py",
    "src/fissionspec/simulator.py",
    "src/fissionspec/statistics.py",
    "src/fissionspec/workload.py",
    "src/fissionspec/workload_generators.py",
)

METRIC_COLUMNS: Final = (
    "schema_version",
    "evidence_class",
    "measurement_warning",
    "claim_boundary",
    "mode",
    "split",
    "workload_kind",
    "cell_id",
    "cluster_id",
    "seed",
    "harness",
    "comparison_reference",
    "policy",
    "requests",
    "output_tokens",
    "makespan_ms",
    "throughput_tokens_per_s",
    "p95_request_latency_ms",
    "p99_tbt_ms",
    "deadline_misses",
    "deadline_miss_rate",
    "target_launches",
    "draft_launches",
    "verifier_slots",
    "padded_verifier_slots",
    "mean_batch",
    "cache_hits",
    "cache_misses",
    "accepted_draft_tokens",
    "max_ready_wait_ms",
    "starved_requests",
    "trace_payload_sha256",
)

FIDELITY_COLUMNS: Final = (
    "schema_version",
    "evidence_class",
    "measurement_warning",
    "claim_boundary",
    "mode",
    "split",
    "workload_kind",
    "cell_id",
    "cluster_id",
    "seed",
    "harness",
    "requests",
    "cache_hits",
    "cache_hit_rate",
    "mean_ttft_ms",
    "p95_ttft_ms",
    "cache_evictions",
    "cache_allocated_pages",
    "cache_peak_allocated_pages",
    "stale_precompute_jobs",
    "precompute_retries",
    "recovery_retries",
    "backpressured_attempts",
    "terminal_failed_jobs",
    "trace_payload_sha256",
)


class StudyIntegrityError(ValueError):
    """Raised when a generated completion-study bundle fails verification."""


@dataclass(frozen=True, slots=True)
class ModeConfig:
    """Predeclared computational budget for a study mode."""

    mode: StudyMode
    seeds: tuple[int, ...]
    request_count: int
    bootstrap_resamples: int
    oracle_jobs: int

    def __post_init__(self) -> None:
        if self.mode not in {"ci", "full"}:
            raise ValueError("mode must be 'ci' or 'full'")
        if len(self.seeds) < 2 or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("mode needs at least two distinct paired seeds")
        for field_name in ("request_count", "bootstrap_resamples", "oracle_jobs"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.bootstrap_resamples < 100:
            raise ValueError("bootstrap_resamples must be at least 100")


def mode_config(mode: StudyMode) -> ModeConfig:
    """Return the frozen CI or full-study computational declaration."""

    if mode == "ci":
        return ModeConfig(
            mode="ci",
            seeds=(0, 1),
            request_count=6,
            bootstrap_resamples=100,
            oracle_jobs=4,
        )
    if mode == "full":
        return ModeConfig(
            mode="full",
            seeds=tuple(range(30)),
            request_count=16,
            bootstrap_resamples=2_000,
            oracle_jobs=6,
        )
    raise ValueError(f"unknown mode: {mode!r}")


@dataclass(frozen=True, slots=True)
class DesignCell:
    """One row of the deterministic fractional-factorial scenario design."""

    cell_id: str
    split: Literal["train", "validation"]
    workload_kind: str
    factor_signs: tuple[int, ...]
    load: Literal["low", "high"]
    capacity: int
    speculation_width: int
    tbt_slo_ms: float
    prompt_tokens: int
    output_tokens: int
    recovery_scale: float
    network_scale: float
    cache_byte_budget: int
    fanout: int
    verifier_slot_ms: float
    controller_setting: Literal["short", "medium", "long"]
    coalesce_ms: float
    horizon2_max_wait_ms: float

    def as_dict(self) -> JsonObject:
        return cast(JsonObject, asdict(self))


_PB12_STARTER: Final = (1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1)
_FACTOR_NAMES: Final = (
    "load",
    "capacity",
    "speculation_width",
    "tbt_slo",
    "prompt_tokens",
    "output_tokens",
    "recovery_scale",
    "network_scale",
    "cache_budget",
    "fanout",
    "physical_slot_cost",
)


def _pb12_rows() -> tuple[tuple[int, ...], ...]:
    """Return a deterministic 12-run, 11-factor Plackett-Burman design."""

    rotations = tuple(
        tuple(_PB12_STARTER[(column - row) % 11] for column in range(11)) for row in range(11)
    )
    rows = (*rotations, (-1,) * 11)
    if any(sum(row[column] for row in rows) != 0 for column in range(11)):
        raise AssertionError("fractional-factorial columns must be balanced")
    return rows


def design_cells() -> tuple[DesignCell, ...]:
    """Materialize all predeclared train/validation scenario cells."""

    cells: list[DesignCell] = []
    controller_levels = (
        ("short", 0.25, 0.5),
        ("medium", 0.75, 1.5),
        ("long", 1.5, 3.0),
    )
    for row_index, signs in enumerate(_pb12_rows()):
        workload_kind = WORKLOAD_KINDS[row_index // 2]
        split: Literal["train", "validation"] = "train" if row_index % 2 == 0 else "validation"
        controller, coalesce, maximum_wait = controller_levels[row_index % 3]
        cells.append(
            DesignCell(
                cell_id=f"pb12-{row_index + 1:02d}-{workload_kind}-{split}",
                split=split,
                workload_kind=workload_kind,
                factor_signs=signs,
                load="high" if signs[0] > 0 else "low",
                capacity=8 if signs[1] > 0 else 4,
                speculation_width=7 if signs[2] > 0 else 3,
                tbt_slo_ms=20.0 if signs[3] > 0 else 50.0,
                prompt_tokens=4096 if signs[4] > 0 else 256,
                output_tokens=32 if signs[5] > 0 else 12,
                recovery_scale=1.6 if signs[6] > 0 else 0.8,
                network_scale=2.0 if signs[7] > 0 else 0.5,
                cache_byte_budget=256 * 1024 if signs[8] > 0 else 8 * 1024 * 1024,
                fanout=3 if signs[9] > 0 else 1,
                verifier_slot_ms=0.04 if signs[10] > 0 else 0.008,
                controller_setting=cast(
                    Literal["short", "medium", "long"],
                    controller,
                ),
                coalesce_ms=coalesce,
                horizon2_max_wait_ms=maximum_wait,
            )
        )
    return tuple(cells)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _profile(cell: DesignCell) -> HardwareProfile:
    recovery = tuple(
        (rows, latency * cell.recovery_scale)
        for rows, latency in ((1, 0.9), (4, 1.3), (8, 2.0), (16, 3.4))
    )
    return HardwareProfile(
        target_curve=LatencyCurve(((1, 1.4), (4, 2.0), (8, 3.0), (16, 5.0))),
        draft_curve=LatencyCurve(((1, 0.35), (4, 0.55), (8, 0.85), (16, 1.4))),
        recovery_curve=LatencyCurve(recovery),
        verifier_slot_ms=cell.verifier_slot_ms,
        name=f"cpu-model-{cell.cell_id}-not-gpu",
    )


def _semantic_probabilities(cell: DesignCell) -> tuple[float, float]:
    hit = 0.86 if cell.cache_byte_budget > 1_000_000 else 0.58
    hit = min(0.95, hit + 0.03 * (cell.fanout - 1))
    acceptance = 0.87 if cell.speculation_width <= 3 else 0.76
    return hit, acceptance


def _arrival_trace(cell: DesignCell, count: int, seed: int) -> ArrivalTrace:
    rng = CounterRNG(f"cpu-completion/arrival/{cell.cell_id}/{seed}")
    if cell.workload_kind == "poisson" or cell.workload_kind == "heterogeneous":
        mean = 0.35 if cell.load == "high" else 1.8
        return poisson_arrivals(
            count=count,
            mean_interarrival_ms=mean,
            rng=rng,
            process_id=cell.cell_id,
        )
    if cell.workload_kind == "mmpp-exact":
        rates = (0.35, 4.0) if cell.load == "high" else (0.12, 1.2)
        return mmpp_arrivals(
            count=count,
            arrival_rates_per_ms=rates,
            transition_rates_per_ms=(0.12, 0.35),
            rng=rng,
            initial_state=seed % 2,
            process_id=cell.cell_id,
        )
    if cell.workload_kind == "pareto-heavy-tail":
        minimum = 0.08 if cell.load == "high" else 0.45
        return pareto_arrivals(
            count=count,
            minimum_interarrival_ms=minimum,
            tail_index=1.35 if cell.load == "high" else 2.2,
            rng=rng,
            process_id=cell.cell_id,
        )
    raise ValueError(f"{cell.workload_kind!r} has no stochastic arrival generator")


def _heterogeneous_workload(
    cell: DesignCell,
    arrivals: ArrivalTrace,
    seed: int,
) -> Workload:
    hit, acceptance = _semantic_probabilities(cell)
    rng = CounterRNG(f"cpu-completion/heterogeneous/{cell.cell_id}/{seed}")
    requests: list[RequestConfig] = []
    for ordinal, arrival in enumerate(arrivals.times_ms):
        prompt_multiplier = (1, 2, 4)[min(2, int(rng.uniform("prompt", ordinal, "length") * 3))]
        output_multiplier = (1, 2)[int(rng.uniform("output", ordinal, "length") >= 0.65)]
        width_delta = -2 if ordinal % 3 == 0 else (2 if ordinal % 3 == 1 else 0)
        width = max(1, cell.speculation_width + width_delta)
        request_hit = min(0.97, max(0.05, hit + (ordinal % 5 - 2) * 0.06))
        request_acceptance = min(
            0.97,
            max(0.05, acceptance + (ordinal % 4 - 1.5) * 0.07),
        )
        requests.append(
            RequestConfig(
                request_id=f"r{ordinal:03d}",
                arrival_ms=arrival,
                output_tokens=cell.output_tokens * output_multiplier,
                prompt_tokens=cell.prompt_tokens * prompt_multiplier,
                speculation_length=width,
                cache_hit_probability=request_hit,
                token_acceptance_probability=request_acceptance,
                tbt_slo_ms=cell.tbt_slo_ms * (0.7 if ordinal % 4 == 0 else 1.0),
            )
        )
    return Workload(tuple(requests), name=f"{cell.cell_id}/seed-{seed}")


def _workload(
    cell: DesignCell,
    config: ModeConfig,
    seed: int,
    replay_trace: Path,
) -> tuple[Workload, dict[str, str]]:
    hit, acceptance = _semantic_probabilities(cell)
    name = f"{cell.cell_id}/seed-{seed}"
    if cell.workload_kind == "synchronized":
        workload = Workload.homogeneous(
            config.request_count,
            arrival_interval_ms=0.0,
            output_tokens=cell.output_tokens,
            speculation_length=cell.speculation_width,
            cache_hit_probability=hit,
            token_acceptance_probability=acceptance,
            tbt_slo_ms=cell.tbt_slo_ms,
            id_prefix="r",
            name=name,
        )
        workload = Workload(
            tuple(
                RequestConfig(
                    request_id=request.request_id,
                    arrival_ms=request.arrival_ms,
                    output_tokens=request.output_tokens,
                    prompt_tokens=cell.prompt_tokens,
                    speculation_length=request.speculation_length,
                    cache_hit_probability=request.cache_hit_probability,
                    token_acceptance_probability=request.token_acceptance_probability,
                    tbt_slo_ms=request.tbt_slo_ms,
                )
                for request in workload
            ),
            name=name,
        )
        return workload, {}
    if cell.workload_kind == "trace-replay":
        loaded = load_trace_csv(
            replay_trace,
            split=cell.split,
            name=name,
        )
        return loaded.workload, {"replay_trace": loaded.source_sha256}
    arrivals = _arrival_trace(cell, config.request_count, seed)
    if cell.workload_kind == "heterogeneous":
        return _heterogeneous_workload(cell, arrivals, seed), {"arrival_trace": arrivals.sha256}
    workload = workload_from_arrivals(
        arrivals,
        name=name,
        output_tokens=cell.output_tokens,
        prompt_tokens=cell.prompt_tokens,
        speculation_length=cell.speculation_width,
        cache_hit_probability=hit,
        token_acceptance_probability=acceptance,
        tbt_slo_ms=cell.tbt_slo_ms,
    )
    return workload, {"arrival_trace": arrivals.sha256}


def _main_policies(cell: DesignCell) -> tuple[object, ...]:
    return (
        SaguaroBarrierPolicy(),
        ImmediateFissionPolicy(),
        FixedCoalescePolicy(coalesce_ms=cell.coalesce_ms),
        FissionSpecPolicy(max_wait_ms=cell.horizon2_max_wait_ms),
    )


def _background_jobs(workload: Workload, cell: DesignCell) -> tuple[BackgroundDraftJob, ...]:
    arrivals = tuple(sorted(request.arrival_ms for request in workload))
    if not arrivals:
        return ()
    count = min(4, max(1, len(arrivals) // 4))
    return tuple(
        BackgroundDraftJob(
            job_id=f"background-{index}",
            release_ms=arrivals[min(index * len(arrivals) // count, len(arrivals) - 1)],
            duration_ms=(0.25 + 0.1 * index) * cell.network_scale,
        )
        for index in range(count)
    )


def _baseline_costs(profile: HardwareProfile, cell: DesignCell) -> BaselineCostModel:
    return BaselineCostModel(
        hardware=profile,
        draft_context_token_ms=0.000015 * cell.network_scale,
        draft_token_ms=0.015 * cell.recovery_scale,
        realignment_base_ms=0.12 * cell.network_scale,
        realignment_per_length_ms=0.0015 * cell.network_scale,
        starvation_threshold_ms=max(2.0, cell.tbt_slo_ms),
    )


def _scheduler_policies(
    cell: DesignCell,
    profile: HardwareProfile,
) -> tuple[object, ...]:
    ordinary = profile.recovery_curve.latency_ms(min(4, cell.capacity))
    calibration = SpectreCalibration(
        ordinary_round_ms=ordinary,
        parallel_round_ms=0.65 * ordinary,
        rollback_penalty_ms=0.9 * ordinary,
    )
    return (
        FIFOScheduler(),
        SPECTREHybridScheduler(
            calibration,
            priority_burst=4,
            context_compression_factor=0.75,
        ),
        EXSpecSlidingPoolScheduler(
            window_size=max(8, 2 * cell.capacity),
            minimum_group_size=2,
        ),
        MyopicSlackScheduler(
            estimated_base_ms=profile.target_curve.latency_ms(1),
            estimated_slot_ms=cell.verifier_slot_ms,
            aging_rate=1.0,
            starvation_bound_ms=max(2.0, cell.tbt_slo_ms),
            max_coalesce_ms=cell.coalesce_ms,
        ),
    )


def _trace_record(
    *,
    trace_kind: str,
    cell: DesignCell,
    cluster_id: str,
    policy: str,
    payload: object,
) -> JsonObject:
    body: JsonObject = {
        "schema_version": SCHEMA_VERSION,
        "evidence_class": "cpu-simulation-model",
        "measurement_warning": WARNING,
        "claim_boundary": CLAIM_BOUNDARY,
        "trace_kind": trace_kind,
        "cell_id": cell.cell_id,
        "split": cell.split,
        "workload_kind": cell.workload_kind,
        "cluster_id": cluster_id,
        "policy": policy,
        "payload": _jsonable(payload),
    }
    return {**body, "payload_sha256": sha256_document(body)}


def _main_metric_row(
    *,
    config: ModeConfig,
    cell: DesignCell,
    cluster_id: str,
    seed: int,
    result: object,
    trace_hash: str,
) -> MetricRow:
    simulation_result = cast("SimulationResult", result)
    metrics = summarize(simulation_result)
    request_by_id = {request.request_id: request for request in simulation_result.workload}
    latencies = [request.latency_ms for request in simulation_result.requests]
    misses = sum(
        request.completion_ms > request_by_id[request.request_id].absolute_deadline_ms
        for request in simulation_result.requests
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_class": "cpu-simulation-model",
        "measurement_warning": WARNING,
        "claim_boundary": CLAIM_BOUNDARY,
        "mode": config.mode,
        "split": cell.split,
        "workload_kind": cell.workload_kind,
        "cell_id": cell.cell_id,
        "cluster_id": cluster_id,
        "seed": seed,
        "harness": MAIN_HARNESS,
        "comparison_reference": MAIN_REFERENCE,
        "policy": simulation_result.policy_name,
        "requests": metrics.requests,
        "output_tokens": metrics.output_tokens,
        "makespan_ms": metrics.makespan_ms,
        "throughput_tokens_per_s": metrics.throughput_tokens_per_s,
        "p95_request_latency_ms": percentile(latencies, 0.95),
        "p99_tbt_ms": metrics.p99_tbt_ms,
        "deadline_misses": misses,
        "deadline_miss_rate": misses / metrics.requests,
        "target_launches": metrics.target_launches,
        "draft_launches": metrics.draft_launches,
        "verifier_slots": sum(
            launch.verifier_slots for launch in simulation_result.target_launches
        ),
        "padded_verifier_slots": metrics.padded_verifier_slots,
        "mean_batch": metrics.mean_batch,
        "cache_hits": metrics.cache_hits,
        "cache_misses": metrics.cache_misses,
        "accepted_draft_tokens": metrics.accepted_draft_tokens,
        "max_ready_wait_ms": 0.0,
        "starved_requests": 0,
        "trace_payload_sha256": trace_hash,
    }


def _scheduler_metric_row(
    *,
    config: ModeConfig,
    cell: DesignCell,
    cluster_id: str,
    seed: int,
    result: BaselineResult,
    trace: PreRealizedTrace,
    trace_hash: str,
) -> MetricRow:
    request_by_id = {request.request_id: request for request in trace.requests}
    latencies = [
        request.completion_ms - request_by_id[request.request_id].arrival_ms
        for request in result.requests
    ]
    output_tokens = sum(len(request.emitted_tokens) for request in result.requests)
    remote_steps = sum(
        step.needs_remote_draft for request in trace.requests for step in request.steps
    )
    all_steps = sum(len(request.steps) for request in trace.requests)
    accepted = sum(step.accepted_length for request in trace.requests for step in request.steps)
    metrics = result.metrics
    throughput = output_tokens * 1000.0 / metrics.makespan_ms
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_class": "cpu-simulation-model",
        "measurement_warning": WARNING,
        "claim_boundary": CLAIM_BOUNDARY,
        "mode": config.mode,
        "split": cell.split,
        "workload_kind": cell.workload_kind,
        "cell_id": cell.cell_id,
        "cluster_id": cluster_id,
        "seed": seed,
        "harness": SCHEDULER_HARNESS,
        "comparison_reference": SCHEDULER_REFERENCE,
        "policy": result.policy_name,
        "requests": len(result.requests),
        "output_tokens": output_tokens,
        "makespan_ms": metrics.makespan_ms,
        "throughput_tokens_per_s": throughput,
        "p95_request_latency_ms": percentile(latencies, 0.95),
        "p99_tbt_ms": 0.0,
        "deadline_misses": metrics.deadline_misses,
        "deadline_miss_rate": metrics.deadline_misses / len(result.requests),
        "target_launches": metrics.target_launches,
        "draft_launches": len(result.draft_launches),
        "verifier_slots": metrics.verifier_slots,
        "padded_verifier_slots": metrics.padded_slots,
        "mean_batch": metrics.mean_real_batch,
        "cache_hits": all_steps - remote_steps,
        "cache_misses": remote_steps,
        "accepted_draft_tokens": accepted,
        "max_ready_wait_ms": metrics.max_ready_wait_ms,
        "starved_requests": metrics.starved_requests,
        "trace_payload_sha256": trace_hash,
    }


def _fidelity_request(
    request: RequestConfig,
    ordinal: int,
) -> FidelityRequest:
    return FidelityRequest(
        request_id=request.request_id,
        arrival_ms=request.arrival_ms,
        prompt_tokens=request.prompt_tokens,
        output_tokens=request.output_tokens,
        speculation_length=request.speculation_length,
        class_weights=(("repetitive", 0.6), ("diffuse", 0.4)),
        correlation_key=f"tenant-{ordinal // 4}" if ordinal % 3 != 0 else None,
        priority=2 if ordinal % 5 == 0 else 0,
    )


def _fidelity_config(cell: DesignCell) -> FidelityConfig:
    costs = ContextCostModel(
        prefill_base_ms=0.15,
        prefill_per_token_ms=0.0004,
        target_base_ms=0.8,
        target_per_row_ms=0.07,
        target_per_context_token_ms=0.00002,
        target_per_verifier_slot_ms=cell.verifier_slot_ms,
        draft_base_ms=0.25,
        recovery_base_ms=0.65 * cell.recovery_scale,
        draft_per_row_ms=0.03,
        draft_per_context_token_ms=0.00001 * cell.recovery_scale,
        draft_per_branch_ms=0.025,
        network_base_ms=0.04 * cell.network_scale,
        network_per_byte_ms=0.0000005 * cell.network_scale,
        network_jitter_ms=0.03 * cell.network_scale,
    )
    return FidelityConfig(
        costs=costs,
        remote=RemoteDraftConfig(
            workers=2 if cell.capacity >= 8 else 1,
            queue_policy="continuous-batching",
            max_batch_size=cell.capacity,
            batch_window_ms=0.08,
            queue_capacity=max(2, cell.capacity // 2),
            failure_probability=0.02 if cell.recovery_scale > 1.0 else 0.0,
            max_retries=1,
            retry_backoff_ms=0.1 * cell.network_scale,
        ),
        cache_byte_budget=cell.cache_byte_budget,
        cache_page_size_bytes=16 * 1024,
        kv_bytes_per_token=4096,
        continuation_tokens=cell.speculation_width,
        fanout=cell.fanout,
        target_batch_size=cell.capacity,
    )


def _run_fidelity(
    *,
    config: ModeConfig,
    cell: DesignCell,
    workload: Workload,
    cluster_id: str,
    seed: int,
) -> tuple[MetricRow, JsonObject]:
    requests = tuple(
        _fidelity_request(request, ordinal) for ordinal, request in enumerate(workload.requests)
    )
    classes = (
        OutcomeClass("repetitive", 0.9, (0.72, 0.2, 0.08)),
        OutcomeClass("diffuse", 0.55, (0.4, 0.34, 0.26)),
    )
    trace = simulate_fidelity_trace(
        requests,
        classes,
        _fidelity_config(cell),
        seed=f"cpu-completion/fidelity/{cell.cell_id}/{seed}",
    )
    record = _trace_record(
        trace_kind=FIDELITY_HARNESS,
        cell=cell,
        cluster_id=cluster_id,
        policy="fidelity-transformer",
        payload=trace,
    )
    trace_hash = cast(str, record["payload_sha256"])
    hits = sum(1 for request in trace.requests if request.cache_hit is True)
    ttft = [request.ttft_ms for request in trace.requests]
    recovery_retries = trace.recovery_trace.retries if trace.recovery_trace else 0
    terminal_failures = len(trace.precompute_trace.terminal_failed_job_ids)
    if trace.recovery_trace:
        terminal_failures += len(trace.recovery_trace.terminal_failed_job_ids)
    backpressured = trace.precompute_trace.backpressured_attempts
    if trace.recovery_trace:
        backpressured += trace.recovery_trace.backpressured_attempts
    row: MetricRow = {
        "schema_version": SCHEMA_VERSION,
        "evidence_class": "cpu-simulation-model",
        "measurement_warning": WARNING,
        "claim_boundary": CLAIM_BOUNDARY,
        "mode": config.mode,
        "split": cell.split,
        "workload_kind": cell.workload_kind,
        "cell_id": cell.cell_id,
        "cluster_id": cluster_id,
        "seed": seed,
        "harness": FIDELITY_HARNESS,
        "requests": len(trace.requests),
        "cache_hits": hits,
        "cache_hit_rate": hits / len(trace.requests),
        "mean_ttft_ms": math.fsum(ttft) / len(ttft),
        "p95_ttft_ms": percentile(ttft, 0.95),
        "cache_evictions": trace.cache_evictions,
        "cache_allocated_pages": trace.cache_allocated_pages,
        "cache_peak_allocated_pages": trace.cache_peak_allocated_pages,
        "stale_precompute_jobs": trace.stale_precompute_jobs,
        "precompute_retries": trace.precompute_trace.retries,
        "recovery_retries": recovery_retries,
        "backpressured_attempts": backpressured,
        "terminal_failed_jobs": terminal_failures,
        "trace_payload_sha256": trace_hash,
    }
    return row, record


def _run_one_cluster(
    *,
    config: ModeConfig,
    cell: DesignCell,
    seed: int,
    replay_trace: Path,
    implementation_digest: str,
) -> tuple[list[MetricRow], MetricRow, list[JsonObject], dict[str, str]]:
    cluster_id = f"{cell.cell_id}/seed-{seed:04d}"
    workload, source_hashes = _workload(cell, config, seed, replay_trace)
    profile = _profile(cell)
    rng_seed = f"cpu-completion/semantic/{cell.cell_id}/{seed}"
    main_results = [
        simulate(
            workload,
            profile,
            cast("SchedulingPolicy", policy),
            CounterRNG(rng_seed),
            max_batch_size=cell.capacity,
        )
        for policy in _main_policies(cell)
    ]
    main_rows: list[MetricRow] = []
    records: list[JsonObject] = []
    immediate_result = None
    for main_result in main_results:
        document = simulation_trace_document(
            main_result,
            implementation_digest=implementation_digest,
            source_hashes=source_hashes,
        )
        record = _trace_record(
            trace_kind=MAIN_HARNESS,
            cell=cell,
            cluster_id=cluster_id,
            policy=main_result.policy_name,
            payload=document,
        )
        records.append(record)
        main_rows.append(
            _main_metric_row(
                config=config,
                cell=cell,
                cluster_id=cluster_id,
                seed=seed,
                result=main_result,
                trace_hash=cast(str, record["payload_sha256"]),
            )
        )
        if main_result.policy_name == "immediate-fission":
            immediate_result = main_result
    if immediate_result is None:
        raise AssertionError("immediate-fission source trace is missing")

    realized = PreRealizedTrace.from_simulation(
        immediate_result,
        background_draft_jobs=_background_jobs(workload, cell),
        name=f"{cluster_id}/pre-realized",
    )
    scheduler_results = [
        DeterministicBaselineSimulator(
            trace=realized,
            scheduler=cast("BaselineScheduler", scheduler),
            costs=_baseline_costs(profile, cell),
            max_batch_size=cell.capacity,
        ).run()
        for scheduler in _scheduler_policies(cell, profile)
    ]
    assert_semantic_equivalence(*scheduler_results)
    for scheduler_result in scheduler_results:
        payload = {
            "result": scheduler_result,
            "source_semantic_signature": realized.semantic_signature,
            "cost_model": _baseline_costs(profile, cell),
        }
        record = _trace_record(
            trace_kind=SCHEDULER_HARNESS,
            cell=cell,
            cluster_id=cluster_id,
            policy=scheduler_result.policy_name,
            payload=payload,
        )
        records.append(record)
        main_rows.append(
            _scheduler_metric_row(
                config=config,
                cell=cell,
                cluster_id=cluster_id,
                seed=seed,
                result=scheduler_result,
                trace=realized,
                trace_hash=cast(str, record["payload_sha256"]),
            )
        )

    fidelity_row, fidelity_record = _run_fidelity(
        config=config,
        cell=cell,
        workload=workload,
        cluster_id=cluster_id,
        seed=seed,
    )
    records.append(fidelity_record)
    return main_rows, fidelity_row, records, source_hashes


def _group_rows(
    rows: Sequence[MetricRow],
) -> dict[tuple[str, str, str], dict[str, MetricRow]]:
    grouped: dict[tuple[str, str, str], dict[str, MetricRow]] = defaultdict(dict)
    for row in rows:
        key = (str(row["harness"]), str(row["cell_id"]), str(row["policy"]))
        cluster = str(row["cluster_id"])
        if cluster in grouped[key]:
            raise AssertionError(f"duplicate cluster row for {key!r}: {cluster}")
        grouped[key][cluster] = row
    return grouped


def _uncertainty_document(
    rows: Sequence[MetricRow],
    fidelity_rows: Sequence[MetricRow],
    config: ModeConfig,
) -> JsonObject:
    grouped = _group_rows(rows)
    cell_by_id = {cell.cell_id: cell for cell in design_cells()}
    headline_ids = tuple(
        f"{cell.cell_id}/{HEADLINE_POLICY}/{metric}"
        for cell in design_cells()
        if cell.split == "validation"
        for metric in (
            "throughput_tokens_per_s",
            "p95_request_latency_ms",
            "deadline_miss_rate",
        )
    )
    multiplicity = bonferroni_metadata(
        headline_ids,
        family_id="validation-h2-vs-barrier",
        familywise_alpha=0.05,
        confirmatory=False,
    )
    directions = {
        "throughput_tokens_per_s": "higher",
        "p95_request_latency_ms": "lower",
        "deadline_miss_rate": "lower",
    }
    comparisons: list[JsonObject] = []
    harness_declarations = (
        (
            MAIN_HARNESS,
            MAIN_REFERENCE,
            (
                "immediate-fission",
                "fixed-coalesce",
                HEADLINE_POLICY,
            ),
        ),
        (
            SCHEDULER_HARNESS,
            SCHEDULER_REFERENCE,
            (
                "spectre-hybrid-abstraction",
                "exspec-sliding-pool-abstraction",
                "myopic-slack-aging-abstraction",
            ),
        ),
    )
    for harness, reference, candidates in harness_declarations:
        for cell in design_cells():
            baseline = grouped[(harness, cell.cell_id, reference)]
            for candidate in candidates:
                candidate_rows = grouped[(harness, cell.cell_id, candidate)]
                if set(candidate_rows) != set(baseline):
                    raise AssertionError("paired comparison has mismatched clusters")
                clusters = tuple(sorted(baseline))
                for metric, direction in directions.items():
                    candidate_values = [float(candidate_rows[key][metric]) for key in clusters]
                    baseline_values = [float(baseline[key][metric]) for key in clusters]
                    orientation = 1.0 if direction == "higher" else -1.0
                    differences = {
                        key: (
                            orientation
                            * (float(candidate_rows[key][metric]) - float(baseline[key][metric])),
                        )
                        for key in clusters
                    }
                    is_headline = (
                        harness == MAIN_HARNESS
                        and candidate == HEADLINE_POLICY
                        and cell.split == "validation"
                    )
                    confidence = (
                        multiplicity.simultaneous_per_hypothesis_confidence_level
                        if is_headline
                        else 0.95
                    )
                    interval = paired_cluster_bootstrap(
                        differences,
                        confidence_level=confidence,
                        resamples=config.bootstrap_resamples,
                        seed=(
                            f"cpu-completion/bootstrap/{config.mode}/{harness}/"
                            f"{cell.cell_id}/{candidate}/{metric}"
                        ),
                    )
                    effect = paired_effect_size(
                        candidate_values,
                        baseline_values,
                        direction=cast(Literal["higher", "lower"], direction),
                    )
                    comparisons.append(
                        {
                            "hypothesis_id": f"{cell.cell_id}/{candidate}/{metric}",
                            "headline": is_headline,
                            "harness": harness,
                            "cell_id": cell.cell_id,
                            "split": cell.split,
                            "candidate": candidate,
                            "reference": reference,
                            "metric": metric,
                            "direction": direction,
                            "clusters": len(clusters),
                            "interval_on_oriented_improvement": interval.as_dict(),
                            "paired_effect": effect.as_dict(),
                        }
                    )

    fidelity_by_cell: dict[str, list[MetricRow]] = defaultdict(list)
    for row in fidelity_rows:
        fidelity_by_cell[str(row["cell_id"])].append(row)
    fidelity_intervals: list[JsonObject] = []
    for cell_id in sorted(fidelity_by_cell):
        cell_rows = sorted(
            fidelity_by_cell[cell_id],
            key=lambda row: str(row["cluster_id"]),
        )
        for metric in ("cache_hit_rate", "p95_ttft_ms", "terminal_failed_jobs"):
            values = {str(row["cluster_id"]): (float(row[metric]),) for row in cell_rows}
            interval = paired_cluster_bootstrap(
                values,
                confidence_level=0.95,
                resamples=config.bootstrap_resamples,
                seed=f"cpu-completion/fidelity-bootstrap/{config.mode}/{cell_id}/{metric}",
            )
            fidelity_intervals.append(
                {
                    "cell_id": cell_id,
                    "split": cell_by_id[cell_id].split,
                    "metric": metric,
                    "clusters": len(values),
                    "interval_on_cluster_mean": interval.as_dict(),
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_class": "cpu-simulation-model-statistics",
        "measurement_warning": WARNING,
        "claim_boundary": CLAIM_BOUNDARY,
        "experimental_unit": "independent paired seed/trace cluster",
        "cross_harness_ranking_permitted": False,
        "multiplicity": multiplicity.as_dict(),
        "comparisons": comparisons,
        "fidelity_cell_intervals": fidelity_intervals,
    }


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _oracle_profile(cell: DesignCell) -> HardwareProfile:
    return HardwareProfile.linear(
        target_overhead_ms=0.8,
        target_per_row_ms=0.18,
        draft_overhead_ms=0.1,
        draft_per_row_ms=0.05,
        recovery_overhead_ms=0.2,
        recovery_per_row_ms=0.07,
        verifier_slot_ms=cell.verifier_slot_ms,
        name=f"oracle-{cell.cell_id}-not-gpu",
    )


def _oracle_problem(cell: DesignCell, job_count: int) -> OracleProblem:
    profile = _oracle_profile(cell)
    widths = tuple(1 + (index * 2 + len(cell.cell_id)) % 3 for index in range(job_count))
    releases = tuple(
        Fraction((index // 2) * (1 if cell.load == "high" else 3), 5) for index in range(job_count)
    )
    jobs = tuple(
        OracleJob(
            job_id=f"j{index}",
            release_time=releases[index],
            width=widths[index],
            deadline=releases[index] + Fraction(5 + (index % 3), 1),
            weight=Fraction(1 + (index % 2), 1),
            cohort_id=f"c{index // 2}",
        )
        for index in range(job_count)
    )
    capacity = OracleCapacity(row_limit=min(3, cell.capacity), slot_limit=9)
    shapes: dict[tuple[int, int], Fraction] = {}
    for subset_mask in range(1, 1 << len(jobs)):
        selected = [jobs[index] for index in range(len(jobs)) if subset_mask & (1 << index)]
        rows = len(selected)
        slots = sum(job.width for job in selected)
        if rows <= capacity.row_limit and slots <= capacity.slot_limit:
            shapes[(rows, slots)] = Fraction(str(profile.target_latency_ms(rows, slots)))
    return OracleProblem(
        jobs,
        capacity,
        ExactLatencySurface(shapes),
        wait=OracleWaitConfig(
            include_release_times=True,
            include_deadline_safe_times=False,
        ),
    )


def _h2_oracle_schedule(problem: OracleProblem, cell: DesignCell) -> ScheduleEvaluation:
    """Evaluate H2 over one-shot exact jobs without using future outcomes."""

    remaining = set(range(len(problem.jobs)))
    time_now = problem.start_time
    events: list[DispatchEvent | WaitEvent] = []
    completions: dict[str, Fraction] = {}
    policy = FissionSpecPolicy(max_wait_ms=cell.horizon2_max_wait_ms)
    profile = _oracle_profile(cell)
    iterations = 0
    while remaining:
        iterations += 1
        if iterations > 10_000:
            raise AssertionError("H2 oracle-gap evaluator did not make progress")
        ready = sorted(
            (index for index in remaining if problem.jobs[index].release_time <= time_now),
            key=lambda index: (
                problem.jobs[index].deadline,
                problem.jobs[index].job_id,
            ),
        )
        if not ready:
            wake = min(problem.jobs[index].release_time for index in remaining)
            events.append(WaitEvent(time_now, wake, WaitKind.FORCED_RELEASE))
            time_now = wake
            continue
        selected: list[int] = []
        slots = 0
        for index in ready:
            job = problem.jobs[index]
            if (
                len(selected) < problem.capacity.row_limit
                and slots + job.width <= problem.capacity.slot_limit
            ):
                selected.append(index)
                slots += job.width
        if not selected:
            raise AssertionError("oracle-gap evaluator found no admissible ready row")
        future_release = min(
            (
                problem.jobs[index].release_time
                for index in remaining
                if problem.jobs[index].release_time > time_now
            ),
            default=None,
        )
        future_indices = (
            tuple(
                sorted(
                    (
                        index
                        for index in remaining
                        if problem.jobs[index].release_time == future_release
                    ),
                    key=lambda index: (
                        problem.jobs[index].deadline,
                        problem.jobs[index].job_id,
                    ),
                )
            )
            if future_release is not None
            else ()
        )
        now_float = float(time_now)
        context = DispatchContext(
            now_ms=now_float,
            ready_count=len(selected),
            capacity=problem.capacity.row_limit,
            oldest_ready_ms=min(float(problem.jobs[index].release_time) for index in selected),
            earliest_deadline_ms=min(float(problem.jobs[index].deadline) for index in selected),
            row_slots=tuple(problem.jobs[index].width for index in selected),
            row_deadlines_ms=tuple(float(problem.jobs[index].deadline) for index in selected),
            profile=profile,
            next_ready_time_ms=float(future_release) if future_release is not None else None,
            next_ready_count=len(future_indices),
            earliest_future_deadline_ms=(
                min(float(problem.jobs[index].deadline) for index in future_indices)
                if future_indices
                else None
            ),
            future_row_slots=tuple(problem.jobs[index].width for index in future_indices),
            future_row_deadlines_ms=tuple(
                float(problem.jobs[index].deadline) for index in future_indices
            ),
        )
        dispatch_float = policy.dispatch_at(context)
        dispatch_at = Fraction(str(dispatch_float))
        if math.isclose(
            dispatch_float,
            float(time_now),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            dispatch_at = time_now
        if dispatch_at > time_now:
            if future_release is None or not math.isclose(
                dispatch_float,
                float(future_release),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise AssertionError(
                    "H2 returned an undeclared exact wait point: "
                    f"cell={cell.cell_id}, now={time_now}, dispatch={dispatch_float}, "
                    f"next_release={future_release}"
                )
            dispatch_at = future_release
            events.append(WaitEvent(time_now, dispatch_at, WaitKind.RELEASE))
            time_now = dispatch_at
            continue
        duration = problem.latency.duration(len(selected), slots)
        end = time_now + duration
        job_ids = tuple(problem.jobs[index].job_id for index in selected)
        events.append(
            DispatchEvent(
                start_time=time_now,
                end_time=end,
                job_ids=job_ids,
                rows=len(selected),
                slots=slots,
            )
        )
        for index in selected:
            completions[problem.jobs[index].job_id] = end
        remaining.difference_update(selected)
        time_now = end
    objective = score_completion_times(problem, completions)
    return ScheduleEvaluation(
        input_hash=problem.input_hash,
        events=tuple(events),
        objective=objective,
        completion_times=tuple((job.job_id, completions[job.job_id]) for job in problem.jobs),
    )


def _oracle_document(config: ModeConfig) -> JsonObject:
    rows: list[JsonObject] = []
    for cell in design_cells():
        if cell.split != "validation":
            continue
        problem = _oracle_problem(cell, config.oracle_jobs)
        limits = OracleSearchLimits(
            max_jobs=config.oracle_jobs,
            max_states=250_000,
            max_transitions=2_000_000,
            max_trace_events=128,
        )
        certificate = solve_general_oracle(problem, limits=limits)
        verification = verify_general_oracle_certificate(
            problem,
            certificate,
            max_verifier_nodes=2_000_000,
        )
        h2 = _h2_oracle_schedule(problem, cell)
        edf = work_conserving_edf(problem)
        rows.append(
            {
                "cell_id": cell.cell_id,
                "split": cell.split,
                "jobs": len(problem.jobs),
                "problem_hash": problem.input_hash,
                "certificate_hash": certificate.certificate_hash,
                "certificate": certificate,
                "verification": verification,
                "h2_schedule": h2,
                "h2_gap": objective_gap(h2.objective, certificate.objective),
                "edf_schedule": edf,
                "edf_gap": objective_gap(edf.objective, certificate.objective),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_class": "exact-cpu-model-oracle",
        "measurement_warning": WARNING,
        "claim_boundary": CLAIM_BOUNDARY,
        "scope": (
            "Bounded one-shot row scheduling only; this is not a full multi-round "
            "serving optimum and not a GPU measurement."
        ),
        "rows": rows,
    }


def _adversarial_costs(
    *,
    target_slot_ms: float = 0.0,
    starvation_ms: float = 50.0,
) -> BaselineCostModel:
    return BaselineCostModel(
        hardware=HardwareProfile.linear(
            target_overhead_ms=0.1,
            target_per_row_ms=0.1,
            draft_overhead_ms=0.01,
            draft_per_row_ms=0.01,
            recovery_overhead_ms=0.01,
            recovery_per_row_ms=0.01,
            verifier_slot_ms=target_slot_ms,
            name="adversarial-cpu-model-not-gpu",
        ),
        draft_context_token_ms=0.0,
        draft_token_ms=0.0,
        realignment_base_ms=0.2,
        realignment_per_length_ms=0.01,
        starvation_threshold_ms=starvation_ms,
    )


def _step(
    width: int,
    accepted: int,
    tokens: tuple[int, ...],
    *,
    remote: bool = False,
    rollback: bool = False,
) -> RealizedStep:
    return RealizedStep(
        speculation_length=width,
        accepted_length=accepted,
        emitted_tokens=tokens,
        needs_remote_draft=remote,
        rollback=rollback,
    )


def _baseline_run(
    trace: PreRealizedTrace,
    scheduler: object,
    costs: BaselineCostModel,
    *,
    batch: int,
) -> BaselineResult:
    return DeterministicBaselineSimulator(
        trace=trace,
        scheduler=cast("BaselineScheduler", scheduler),
        costs=costs,
        max_batch_size=batch,
    ).run()


def _adversarial_document() -> JsonObject:
    spectre_trace = PreRealizedTrace(
        (
            RealizedRequest(
                "wide",
                0.0,
                100.0,
                0,
                (
                    _step(1, 0, (1,), remote=True, rollback=True),
                    _step(64, 0, (2,)),
                ),
            ),
        ),
        name="spectre-width-counterexample",
    )
    spectre_costs = _adversarial_costs(target_slot_ms=0.1)
    parallel = _baseline_run(
        spectre_trace,
        SPECTREHybridScheduler(SpectreCalibration(10.0, 1.0, 1.0)),
        spectre_costs,
        batch=1,
    )
    ordinary = _baseline_run(
        spectre_trace,
        SPECTREHybridScheduler(SpectreCalibration(1.0, 10.0, 1.0)),
        spectre_costs,
        batch=1,
    )
    assert_semantic_equivalence(parallel, ordinary)

    pair_steps = tuple(_step(1, 0, (index,)) for index in range(8))
    exspec_trace = PreRealizedTrace(
        (
            RealizedRequest("unique", 0.0, 100.0, 99, (_step(1, 0, (99,)),)),
            RealizedRequest("pair-a", 0.0, 100.0, 0, pair_steps),
            RealizedRequest("pair-b", 0.0, 100.0, 0, pair_steps),
        ),
        name="exspec-starvation-counterexample",
    )
    exspec_costs = _adversarial_costs(starvation_ms=0.5)
    exspec = _baseline_run(
        exspec_trace,
        EXSpecSlidingPoolScheduler(window_size=3),
        exspec_costs,
        batch=2,
    )
    fifo = _baseline_run(exspec_trace, FIFOScheduler(), exspec_costs, batch=2)
    assert_semantic_equivalence(exspec, fifo)

    aging_trace = PreRealizedTrace(
        (
            RealizedRequest("blocker", 0.0, 0.15, 0, (_step(1, 0, (1,)),)),
            RealizedRequest("old-wide", 0.0, 100.0, 0, (_step(50, 0, (2,)),)),
            RealizedRequest("new-tight", 0.19, 0.7, 0, (_step(1, 0, (3,)),)),
        ),
        name="aging-counterexample",
    )
    aging_costs = _adversarial_costs(target_slot_ms=0.1)
    aging = _baseline_run(
        aging_trace,
        MyopicSlackScheduler(
            estimated_base_ms=0.1,
            estimated_slot_ms=0.1,
            aging_rate=1.0,
            starvation_bound_ms=0.15,
        ),
        aging_costs,
        batch=1,
    )
    slack = _baseline_run(
        aging_trace,
        MyopicSlackScheduler(
            estimated_base_ms=0.1,
            estimated_slot_ms=0.1,
            aging_rate=0.0,
            starvation_bound_ms=1_000.0,
        ),
        aging_costs,
        batch=1,
    )
    assert_semantic_equivalence(aging, slack)
    unique_exspec = next(request for request in exspec.requests if request.request_id == "unique")
    unique_fifo = next(request for request in fifo.requests if request.request_id == "unique")
    tight_aging = next(request for request in aging.requests if request.request_id == "new-tight")
    tight_slack = next(request for request in slack.requests if request.request_id == "new-tight")
    cases = (
        {
            "case_id": "spectre-fixed-threshold-wide-padding",
            "invariant": "semantic outputs remain identical",
            "observed_limitation": (
                "a rollback-ratio threshold chooses parallel mode while physical "
                "wide-row padding increases modeled makespan"
            ),
            "parallel_makespan_ms": parallel.metrics.makespan_ms,
            "ordinary_makespan_ms": ordinary.metrics.makespan_ms,
            "parallel_padded_slots": parallel.metrics.padded_slots,
            "witness": parallel.metrics.makespan_ms > ordinary.metrics.makespan_ms,
        },
        {
            "case_id": "exspec-homogeneous-group-starvation",
            "invariant": "semantic outputs remain identical",
            "observed_limitation": ("homogeneous grouping delays an old unique-length request"),
            "exspec_unique_wait_ms": unique_exspec.max_ready_wait_ms,
            "fifo_unique_wait_ms": unique_fifo.max_ready_wait_ms,
            "witness": unique_exspec.max_ready_wait_ms > unique_fifo.max_ready_wait_ms,
        },
        {
            "case_id": "myopic-aging-tight-deadline-inversion",
            "invariant": "semantic outputs remain identical",
            "observed_limitation": (
                "fairness promotion admits a wide old row before a new tight row"
            ),
            "aging_tight_deadline_missed": tight_aging.deadline_missed,
            "slack_only_tight_deadline_missed": tight_slack.deadline_missed,
            "witness": tight_aging.deadline_missed and not tight_slack.deadline_missed,
        },
    )
    if not all(bool(case["witness"]) for case in cases):
        raise AssertionError("a predeclared adversarial witness stopped reproducing")
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_class": "deterministic-cpu-model-counterexamples",
        "measurement_warning": WARNING,
        "claim_boundary": CLAIM_BOUNDARY,
        "cases": cases,
    }


def _jsonable(value: object) -> object:
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "+Infinity" if value > 0.0 else "-Infinity"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _csv_bytes(rows: Sequence[MetricRow], columns: Sequence[str]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=list(columns),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return output.getvalue().encode("utf-8")


def _gzip_jsonl(records: Iterable[JsonObject]) -> bytes:
    raw = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        fileobj=raw,
        mtime=0,
        compresslevel=9,
    ) as compressed:
        for record in records:
            compressed.write(canonical_json_bytes(record))
    return raw.getvalue()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _artifact_entry(name: str, payload: bytes) -> JsonObject:
    return {
        "path": name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "evidence_class": "cpu-simulation-model",
        "measurement_warning": WARNING,
    }


def _summary_markdown(
    config: ModeConfig,
    metrics: Sequence[MetricRow],
    fidelity: Sequence[MetricRow],
    oracle: JsonObject,
    uncertainty: JsonObject,
    adversarial: JsonObject,
) -> bytes:
    cell_by_id = {cell.cell_id: cell for cell in design_cells()}
    raw_comparisons = uncertainty["comparisons"]
    if not isinstance(raw_comparisons, list):
        raise AssertionError("uncertainty comparisons must be a list")
    headline = [
        cast(JsonObject, comparison)
        for comparison in raw_comparisons
        if isinstance(comparison, dict) and comparison.get("headline") is True
    ]
    metric_labels = {
        "throughput_tokens_per_s": "throughput tokens/s",
        "p95_request_latency_ms": "p95 request latency ms",
        "deadline_miss_rate": "deadline-miss fraction",
    }
    lines = [
        "# FissionSpec GPU-free completion study",
        "",
        f"> **{WARNING}**",
        "",
        CLAIM_BOUNDARY,
        "",
        "## Bundle scope",
        "",
        f"- Mode: `{config.mode}`",
        f"- Fractional-factorial cells: {len(design_cells())}",
        f"- Independent paired seed/trace clusters per cell: {len(config.seeds)}",
        f"- Policy rows: {len(metrics)}",
        f"- Fidelity rows: {len(fidelity)}",
        f"- Exact bounded oracle cells: {len(cast(list[object], oracle['rows']))}",
        "",
        "The policy table has two non-comparable strata. Decoder-policy rows may be",
        "compared only with `saguaro-barrier`; scheduler-abstraction rows may be",
        "compared only with `fifo-ordinary-reference`. The study never ranks a",
        "policy from one harness against a policy from the other.",
        "",
        "Validation headline intervals use paired seed/trace clusters and a",
        "Bonferroni-declared family. Training cells and non-headline policies are",
        "exploratory. `uncertainty.json` contains effect sizes, interval metadata,",
        "RNG provenance, and resample fingerprints.",
        "",
        "## Validation headline: H2 versus barrier",
        "",
        "Positive values mean oriented improvement: more throughput, lower latency,",
        "or fewer misses. Intervals are simultaneous within the predeclared",
        "validation family. They quantify paired simulator-seed variation only.",
        "",
        "| workload | metric | paired mean improvement | simultaneous interval | clusters |",
        "|---|---|---:|---:|---:|",
    ]
    for comparison in headline:
        cell_id = str(comparison["cell_id"])
        metric = str(comparison["metric"])
        interval = comparison["interval_on_oriented_improvement"]
        if not isinstance(interval, dict):
            raise AssertionError("headline interval must be an object")
        point = float(interval["point_estimate"])
        lower = float(interval["lower"])
        upper = float(interval["upper"])
        lines.append(
            f"| {cell_by_id[cell_id].workload_kind} | {metric_labels[metric]} | "
            f"{point:+.6g} | [{lower:+.6g}, {upper:+.6g}] | "
            f"{cast(int, comparison['clusters'])} |"
        )
    fidelity_hit_rates = [float(row["cache_hit_rate"]) for row in fidelity]
    fidelity_ttft = [float(row["p95_ttft_ms"]) for row in fidelity]
    raw_oracle_rows = oracle["rows"]
    raw_cases = adversarial["cases"]
    if not isinstance(raw_oracle_rows, list) or not isinstance(raw_cases, tuple):
        raise AssertionError("summary inputs have invalid oracle/adversarial rows")
    lines.extend(
        [
            "",
            "These model outcomes do not support universal H2 dominance: the signed",
            "effect changes across validation workloads. That negative result is",
            "preserved rather than filtered.",
            "",
            "## Fidelity and bounded exact checks",
            "",
            f"- Fidelity cache-hit observations span "
            f"`{min(fidelity_hit_rates):.6g}` to `{max(fidelity_hit_rates):.6g}`.",
            f"- Fidelity p95 TTFT observations span "
            f"`{min(fidelity_ttft):.6g}` to `{max(fidelity_ttft):.6g}` ms.",
            f"- Exact generalized-oracle certificates: {len(raw_oracle_rows)}.",
            f"- Deterministic adversarial witnesses reproduced: {len(raw_cases)}.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "PYTHONPATH=src python experiments/run_cpu_completion_study.py --mode full",
            "PYTHONPATH=src python experiments/run_cpu_completion_study.py --verify-only \\",
            "  experiments/results/cpu_completion_full",
            "```",
            "",
            "Every event/request trace is retained in deterministic `traces.jsonl.gz`.",
            "The manifest hashes every artifact. Wall-clock runtime is printed by the",
            "driver but excluded from the bundle so golden reruns are byte-identical.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


@dataclass(frozen=True, slots=True)
class StudyRun:
    output_dir: Path
    manifest_sha256: str
    metric_rows: int
    fidelity_rows: int
    trace_records: int
    elapsed_seconds: float


def run_study(
    *,
    mode: StudyMode,
    output_dir: Path,
    replay_trace: Path | None = None,
) -> StudyRun:
    """Execute the predeclared study and atomically write a verified bundle."""

    started = time.monotonic()
    config = mode_config(mode)
    root = _repo_root()
    replay = (
        root / "configs/replay_trace.example.csv"
        if replay_trace is None
        else replay_trace.resolve()
    )
    implementation_digest = implementation_sha256(root, IMPLEMENTATION_PATHS)
    metric_rows: list[MetricRow] = []
    fidelity_rows: list[MetricRow] = []
    trace_records: list[JsonObject] = []
    source_hashes: dict[str, str] = {
        "replay_trace": hashlib.sha256(replay.read_bytes()).hexdigest()
    }
    for cell in design_cells():
        for seed in config.seeds:
            rows, fidelity, records, cluster_hashes = _run_one_cluster(
                config=config,
                cell=cell,
                seed=seed,
                replay_trace=replay,
                implementation_digest=implementation_digest,
            )
            metric_rows.extend(rows)
            fidelity_rows.append(fidelity)
            trace_records.extend(records)
            for label, digest in cluster_hashes.items():
                source_hashes[f"{cell.cell_id}/{seed}/{label}"] = digest

    metric_rows.sort(
        key=lambda row: (
            str(row["harness"]),
            str(row["cell_id"]),
            int(row["seed"]),
            str(row["policy"]),
        )
    )
    fidelity_rows.sort(key=lambda row: (str(row["cell_id"]), int(row["seed"])))
    trace_records.sort(
        key=lambda record: (
            str(record["trace_kind"]),
            str(record["cell_id"]),
            str(record["cluster_id"]),
            str(record["policy"]),
        )
    )
    uncertainty = _uncertainty_document(metric_rows, fidelity_rows, config)
    oracle = _oracle_document(config)
    adversarial = _adversarial_document()
    design: JsonObject = {
        "schema_version": SCHEMA_VERSION,
        "evidence_class": "cpu-simulation-model-design",
        "measurement_warning": WARNING,
        "claim_boundary": CLAIM_BOUNDARY,
        "mode": asdict(config),
        "fractional_factorial": {
            "kind": "12-run Plackett-Burman screening design",
            "factor_names": _FACTOR_NAMES,
            "controller_block": (
                "deterministic three-level short/medium/long block; not claimed orthogonal"
            ),
            "cells": [cell.as_dict() for cell in design_cells()],
        },
        "train_validation_policy": (
            "Each workload family has one train and one validation scenario; "
            "validation cells alone enter the declared headline family."
        ),
        "harness_strata": {
            MAIN_HARNESS: {
                "reference": MAIN_REFERENCE,
                "policies": (
                    MAIN_REFERENCE,
                    "immediate-fission",
                    "fixed-coalesce",
                    HEADLINE_POLICY,
                ),
            },
            SCHEDULER_HARNESS: {
                "reference": SCHEDULER_REFERENCE,
                "policies": (
                    SCHEDULER_REFERENCE,
                    "spectre-hybrid-abstraction",
                    "exspec-sliding-pool-abstraction",
                    "myopic-slack-aging-abstraction",
                ),
            },
            FIDELITY_HARNESS: {
                "reference": None,
                "policies": ("fidelity-transformer",),
            },
        },
        "cross_harness_ranking_permitted": False,
        "implementation_sha256": implementation_digest,
        "source_sha256": dict(sorted(source_hashes.items())),
    }
    environment: JsonObject = {
        "schema_version": SCHEMA_VERSION,
        "evidence_class": "cpu-simulation-environment",
        "measurement_warning": WARNING,
        "claim_boundary": CLAIM_BOUNDARY,
        "environment": environment_manifest(implementation_digest=implementation_digest),
    }

    payloads = {
        "design.json": canonical_json_bytes(_jsonable(design)),
        "metrics.csv": _csv_bytes(metric_rows, METRIC_COLUMNS),
        "fidelity_metrics.csv": _csv_bytes(fidelity_rows, FIDELITY_COLUMNS),
        "uncertainty.json": canonical_json_bytes(_jsonable(uncertainty)),
        "oracle.json": canonical_json_bytes(_jsonable(oracle)),
        "adversarial.json": canonical_json_bytes(_jsonable(adversarial)),
        "traces.jsonl.gz": _gzip_jsonl(trace_records),
        "environment.json": canonical_json_bytes(_jsonable(environment)),
        "SUMMARY.md": _summary_markdown(
            config,
            metric_rows,
            fidelity_rows,
            oracle,
            uncertainty,
            adversarial,
        ),
    }
    output = output_dir.resolve()
    for name, payload in payloads.items():
        _atomic_write(output / name, payload)
    manifest_payload: JsonObject = {
        "schema_version": SCHEMA_VERSION,
        "evidence_class": "cpu-simulation-model-bundle",
        "measurement_warning": WARNING,
        "claim_boundary": CLAIM_BOUNDARY,
        "mode": mode,
        "cross_harness_ranking_permitted": False,
        "implementation_sha256": implementation_digest,
        "artifacts": [_artifact_entry(name, payload) for name, payload in sorted(payloads.items())],
    }
    manifest: JsonObject = {
        **manifest_payload,
        "payload_sha256": sha256_document(manifest_payload),
    }
    manifest_bytes = canonical_json_bytes(manifest)
    _atomic_write(output / "manifest.json", manifest_bytes)
    verified = verify_bundle(output)
    return StudyRun(
        output_dir=output,
        manifest_sha256=verified,
        metric_rows=len(metric_rows),
        fidelity_rows=len(fidelity_rows),
        trace_records=len(trace_records),
        elapsed_seconds=time.monotonic() - started,
    )


def _safe_artifact_path(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise StudyIntegrityError("artifact paths must be non-empty and relative")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise StudyIntegrityError("artifact path escapes bundle root") from error
    return resolved


def verify_bundle(output_dir: Path) -> str:
    """Verify the manifest envelope and every exact artifact byte string."""

    root = output_dir.resolve()
    manifest_path = root / "manifest.json"
    try:
        document = json.loads(manifest_path.read_bytes())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise StudyIntegrityError("manifest is not valid UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise StudyIntegrityError("manifest root must be an object")
    supplied = document.get("payload_sha256")
    if not isinstance(supplied, str):
        raise StudyIntegrityError("manifest is missing payload_sha256")
    payload = dict(document)
    payload.pop("payload_sha256", None)
    if payload.get("measurement_warning") != WARNING:
        raise StudyIntegrityError("manifest lost its not-GPU warning")
    if payload.get("cross_harness_ranking_permitted") is not False:
        raise StudyIntegrityError("manifest permits an invalid cross-harness ranking")
    actual = sha256_document(payload)
    if actual != supplied:
        raise StudyIntegrityError("manifest payload hash mismatch")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise StudyIntegrityError("manifest has no artifacts")
    seen: set[str] = set()
    for raw_entry in artifacts:
        if not isinstance(raw_entry, dict):
            raise StudyIntegrityError("artifact entry must be an object")
        relative = raw_entry.get("path")
        if not isinstance(relative, str) or relative in seen:
            raise StudyIntegrityError("artifact paths must be unique strings")
        seen.add(relative)
        if raw_entry.get("measurement_warning") != WARNING:
            raise StudyIntegrityError("artifact manifest entry lost its warning")
        artifact = _safe_artifact_path(root, relative)
        try:
            contents = artifact.read_bytes()
        except OSError as error:
            raise StudyIntegrityError(f"cannot read artifact {relative!r}") from error
        if raw_entry.get("bytes") != len(contents):
            raise StudyIntegrityError(f"artifact byte count mismatch: {relative}")
        if raw_entry.get("sha256") != hashlib.sha256(contents).hexdigest():
            raise StudyIntegrityError(f"artifact hash mismatch: {relative}")
    return actual


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("ci", "full"),
        default="ci",
        help="ci is a fast structural check; full predeclares 30 paired clusters per cell",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=("bundle directory (default: experiments/results/cpu_completion_<mode>)"),
    )
    parser.add_argument(
        "--replay-trace",
        type=Path,
        help="replay CSV (default: configs/replay_trace.example.csv)",
    )
    parser.add_argument(
        "--verify-only",
        type=Path,
        help="verify an existing bundle instead of running the study",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.verify_only is not None:
        digest = verify_bundle(args.verify_only)
        print(f"verified manifest payload sha256={digest}")
        return 0
    mode = cast(StudyMode, args.mode)
    destination = (
        _repo_root() / f"experiments/results/cpu_completion_{mode}"
        if args.output_dir is None
        else args.output_dir
    )
    result = run_study(
        mode=mode,
        output_dir=destination,
        replay_trace=args.replay_trace,
    )
    print(
        f"wrote {result.output_dir} with {result.metric_rows} policy rows, "
        f"{result.fidelity_rows} fidelity rows, {result.trace_records} traces; "
        f"manifest={result.manifest_sha256}; runtime={result.elapsed_seconds:.3f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
