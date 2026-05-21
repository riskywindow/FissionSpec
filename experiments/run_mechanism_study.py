#!/usr/bin/env python3
"""Run the frozen CPU-only FissionSpec causal mechanism study.

The experiment has two deliberately separate evidence strata:

* a decoder-policy simulator stratum for recovery latency, physical verifier
  slot cost, target batch capacity, and controller maximum wait; and
* a one-round fidelity stratum for cache budget, branch fanout, network jitter,
  and remote-worker failure probability.

Every contrast changes exactly one declared configuration field relative to a
shared reference.  Seed clusters, arrivals, latent request classes, acceptance
draws, cache outcomes, network jitter draws, and failure draws are paired by
counter-addressed RNG keys.  The output is synthetic CPU evidence, never a GPU
measurement or a calibrated production-performance claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
from dataclasses import asdict, dataclass, is_dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Final, Literal, TypeAlias, cast

from fissionspec.fidelity import (
    ContextCostModel,
    FidelityConfig,
    FidelityRequest,
    FidelityTrace,
    OutcomeClass,
    RemoteDraftConfig,
    simulate_fidelity_trace,
)
from fissionspec.metrics import summarize
from fissionspec.policies import FissionSpecPolicy
from fissionspec.profiles import HardwareProfile, LatencyCurve
from fissionspec.rng import CounterRNG
from fissionspec.simulator import simulate
from fissionspec.workload import RequestConfig, Workload
from fissionspec.workload_generators import poisson_arrivals

SCHEMA_VERSION: Final = 1
WARNING: Final = "SYNTHETIC CPU MODEL / NOT A GPU MEASUREMENT"
CLAIM_BOUNDARY: Final = (
    "These paired effects identify interventions inside the declared CPU models only. "
    "They do not estimate CUDA-kernel, accelerator-memory, network-fabric, power, "
    "or production-serving effects."
)
FAMILYWISE_ALPHA: Final = 0.05
READINESS_RESTRICTION_MS: Final = 25.0
StudyMode = Literal["ci", "full"]
JsonObject: TypeAlias = dict[str, object]
MetricValue: TypeAlias = str | int | float
MetricRow: TypeAlias = dict[str, MetricValue]

DECODER_CONFIRMATORY_METRICS: Final = (
    "throughput_tokens_per_s",
    "p95_tbt_ms",
    "target_launches_per_request",
)
FIDELITY_CONFIRMATORY_METRICS: Final = (
    "cache_hit_rate",
    "restricted_next_ready_delay_ms",
    "next_round_unready_rate",
)
DESCRIPTIVE_METRICS: Final = {
    "decoder-policy": (
        "request_slo_miss_rate",
        "mean_batch",
        "draft_launches_per_request",
    ),
    "one-round-fidelity": (
        "mean_ttft_ms",
        "ready_within_5ms_rate",
        "remote_attempts_per_request",
        "cache_evictions_per_request",
    ),
}
IMPLEMENTATION_PATHS: Final = (
    "experiments/run_mechanism_study.py",
    "src/fissionspec/fidelity.py",
    "src/fissionspec/metrics.py",
    "src/fissionspec/policies.py",
    "src/fissionspec/profiles.py",
    "src/fissionspec/rng.py",
    "src/fissionspec/simulator.py",
    "src/fissionspec/workload.py",
    "src/fissionspec/workload_generators.py",
)


class MechanismStudyError(ValueError):
    """Raised when the frozen study or one of its artifacts is inconsistent."""


@dataclass(frozen=True, slots=True)
class ModeConfig:
    """Finite computation budget for CI and checked-in full modes."""

    mode: StudyMode
    seeds: tuple[int, ...]
    decoder_requests: int
    fidelity_requests: int
    bootstrap_resamples: int


def mode_config(mode: StudyMode) -> ModeConfig:
    """Return the predeclared computation budget."""

    if mode == "ci":
        return ModeConfig(
            mode="ci",
            seeds=(0, 1, 2, 3),
            decoder_requests=12,
            fidelity_requests=12,
            bootstrap_resamples=500,
        )
    if mode == "full":
        return ModeConfig(
            mode="full",
            seeds=tuple(range(30)),
            decoder_requests=48,
            fidelity_requests=48,
            bootstrap_resamples=20_000,
        )
    raise ValueError(f"unknown mode: {mode!r}")


@dataclass(frozen=True, slots=True)
class DecoderSetting:
    """Flat decoder configuration; interventions replace one field."""

    recovery_latency_scale: float = 1.0
    verifier_slot_ms: float = 0.018
    batch_capacity: int = 16
    controller_max_wait_ms: float = 2.0


@dataclass(frozen=True, slots=True)
class FidelitySetting:
    """Flat one-round fidelity configuration; interventions replace one field."""

    cache_byte_budget: int = 1 * 1024 * 1024
    fanout: int = 2
    network_jitter_ms: float = 0.03
    network_failure_probability: float = 0.02


@dataclass(frozen=True, slots=True)
class Intervention:
    """One one-factor-at-a-time setting relative to a shared reference."""

    stratum: Literal["decoder-policy", "one-round-fidelity"]
    factor: str
    level: Literal["lower", "upper"]
    setting_id: str
    changed_field: str
    reference_value: int | float
    intervention_value: int | float


def decoder_reference() -> DecoderSetting:
    """Return the immutable decoder reference setting."""

    return DecoderSetting()


def fidelity_reference() -> FidelitySetting:
    """Return the immutable fidelity reference setting."""

    return FidelitySetting()


def decoder_interventions() -> tuple[Intervention, ...]:
    """Return the frozen decoder-policy OFAT contrasts."""

    return (
        Intervention(
            "decoder-policy",
            "recovery-latency",
            "lower",
            "decoder/recovery-latency/lower",
            "recovery_latency_scale",
            1.0,
            0.5,
        ),
        Intervention(
            "decoder-policy",
            "recovery-latency",
            "upper",
            "decoder/recovery-latency/upper",
            "recovery_latency_scale",
            1.0,
            2.0,
        ),
        Intervention(
            "decoder-policy",
            "physical-verifier-slot-cost",
            "lower",
            "decoder/physical-verifier-slot-cost/lower",
            "verifier_slot_ms",
            0.018,
            0.0,
        ),
        Intervention(
            "decoder-policy",
            "physical-verifier-slot-cost",
            "upper",
            "decoder/physical-verifier-slot-cost/upper",
            "verifier_slot_ms",
            0.018,
            0.06,
        ),
        Intervention(
            "decoder-policy",
            "target-batch-capacity",
            "lower",
            "decoder/target-batch-capacity/lower",
            "batch_capacity",
            16,
            8,
        ),
        Intervention(
            "decoder-policy",
            "target-batch-capacity",
            "upper",
            "decoder/target-batch-capacity/upper",
            "batch_capacity",
            16,
            32,
        ),
        Intervention(
            "decoder-policy",
            "controller-max-wait",
            "lower",
            "decoder/controller-max-wait/lower",
            "controller_max_wait_ms",
            2.0,
            0.0,
        ),
        Intervention(
            "decoder-policy",
            "controller-max-wait",
            "upper",
            "decoder/controller-max-wait/upper",
            "controller_max_wait_ms",
            2.0,
            5.0,
        ),
    )


def fidelity_interventions() -> tuple[Intervention, ...]:
    """Return the frozen one-round-fidelity OFAT contrasts."""

    return (
        Intervention(
            "one-round-fidelity",
            "cache-budget",
            "lower",
            "fidelity/cache-budget/lower",
            "cache_byte_budget",
            1 * 1024 * 1024,
            256 * 1024,
        ),
        Intervention(
            "one-round-fidelity",
            "cache-budget",
            "upper",
            "fidelity/cache-budget/upper",
            "cache_byte_budget",
            1 * 1024 * 1024,
            8 * 1024 * 1024,
        ),
        Intervention(
            "one-round-fidelity",
            "branch-fanout",
            "lower",
            "fidelity/branch-fanout/lower",
            "fanout",
            2,
            1,
        ),
        Intervention(
            "one-round-fidelity",
            "branch-fanout",
            "upper",
            "fidelity/branch-fanout/upper",
            "fanout",
            2,
            3,
        ),
        Intervention(
            "one-round-fidelity",
            "network-jitter",
            "lower",
            "fidelity/network-jitter/lower",
            "network_jitter_ms",
            0.03,
            0.0,
        ),
        Intervention(
            "one-round-fidelity",
            "network-jitter",
            "upper",
            "fidelity/network-jitter/upper",
            "network_jitter_ms",
            0.03,
            0.2,
        ),
        Intervention(
            "one-round-fidelity",
            "network-failure",
            "lower",
            "fidelity/network-failure/lower",
            "network_failure_probability",
            0.02,
            0.0,
        ),
        Intervention(
            "one-round-fidelity",
            "network-failure",
            "upper",
            "fidelity/network-failure/upper",
            "network_failure_probability",
            0.02,
            0.15,
        ),
    )


def all_interventions() -> tuple[Intervention, ...]:
    """Return all predeclared contrasts in stable order."""

    return decoder_interventions() + fidelity_interventions()


def _apply_decoder(intervention: Intervention) -> DecoderSetting:
    if intervention.stratum != "decoder-policy":
        raise MechanismStudyError("decoder intervention has the wrong stratum")
    reference = decoder_reference()
    if intervention.changed_field == "recovery_latency_scale":
        setting = replace(
            reference,
            recovery_latency_scale=float(intervention.intervention_value),
        )
    elif intervention.changed_field == "verifier_slot_ms":
        setting = replace(
            reference,
            verifier_slot_ms=float(intervention.intervention_value),
        )
    elif intervention.changed_field == "batch_capacity":
        setting = replace(
            reference,
            batch_capacity=int(intervention.intervention_value),
        )
    elif intervention.changed_field == "controller_max_wait_ms":
        setting = replace(
            reference,
            controller_max_wait_ms=float(intervention.intervention_value),
        )
    else:
        raise MechanismStudyError(f"unknown decoder field {intervention.changed_field!r}")
    _assert_one_field_change(decoder_reference(), setting, intervention)
    return setting


def _apply_fidelity(intervention: Intervention) -> FidelitySetting:
    if intervention.stratum != "one-round-fidelity":
        raise MechanismStudyError("fidelity intervention has the wrong stratum")
    reference = fidelity_reference()
    if intervention.changed_field == "cache_byte_budget":
        setting = replace(
            reference,
            cache_byte_budget=int(intervention.intervention_value),
        )
    elif intervention.changed_field == "fanout":
        setting = replace(
            reference,
            fanout=int(intervention.intervention_value),
        )
    elif intervention.changed_field == "network_jitter_ms":
        setting = replace(
            reference,
            network_jitter_ms=float(intervention.intervention_value),
        )
    elif intervention.changed_field == "network_failure_probability":
        setting = replace(
            reference,
            network_failure_probability=float(intervention.intervention_value),
        )
    else:
        raise MechanismStudyError(f"unknown fidelity field {intervention.changed_field!r}")
    _assert_one_field_change(fidelity_reference(), setting, intervention)
    return setting


def _assert_one_field_change(
    reference: DecoderSetting | FidelitySetting,
    intervention_setting: DecoderSetting | FidelitySetting,
    intervention: Intervention,
) -> None:
    reference_values = asdict(reference)
    intervention_values = asdict(intervention_setting)
    changed = tuple(
        field for field in reference_values if reference_values[field] != intervention_values[field]
    )
    if changed != (intervention.changed_field,):
        raise MechanismStudyError(
            f"{intervention.setting_id} changes fields {changed}, expected "
            f"{intervention.changed_field!r}"
        )
    if reference_values[intervention.changed_field] != intervention.reference_value:
        raise MechanismStudyError(f"{intervention.setting_id} reference value drifted")
    if intervention_values[intervention.changed_field] != intervention.intervention_value:
        raise MechanismStudyError(f"{intervention.setting_id} intervention value drifted")


def _canonical(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _canonical(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"cannot canonically encode {type(value).__name__}")


def canonical_json_bytes(value: object) -> bytes:
    """Return the study's stable JSON encoding."""

    return json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_document(value: object) -> str:
    """Hash a canonical JSON document."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _implementation_sha256(repo_root: Path) -> tuple[str, dict[str, str]]:
    source_hashes: dict[str, str] = {}
    aggregate = hashlib.sha256()
    for relative in IMPLEMENTATION_PATHS:
        payload = (repo_root / relative).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        source_hashes[relative] = digest
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(bytes.fromhex(digest))
    return aggregate.hexdigest(), source_hashes


def _decoder_profile(setting: DecoderSetting) -> HardwareProfile:
    reference = HardwareProfile()
    return HardwareProfile(
        target_curve=reference.target_curve,
        draft_curve=reference.draft_curve,
        recovery_curve=LatencyCurve(
            tuple(
                (rows, latency_ms * setting.recovery_latency_scale)
                for rows, latency_ms in reference.recovery_curve.points
            )
        ),
        verifier_slot_ms=setting.verifier_slot_ms,
        name=f"mechanism-decoder/{sha256_document(asdict(setting))[:12]}/not-gpu",
    )


def _decoder_workload(seed: int, request_count: int) -> Workload:
    if request_count < 4:
        raise ValueError("decoder mechanism workload needs at least four requests")
    pair_count = min(8, request_count // 3)
    timing_rng = CounterRNG(f"mechanism-study/decoder-arrivals/{seed}")
    requests: list[RequestConfig] = []
    for pair in range(pair_count):
        base_ms = pair * 25.0 + 0.2 * timing_rng.uniform(
            f"pair-{pair}",
            0,
            "base-jitter",
        )
        requests.extend(
            (
                RequestConfig(
                    request_id=f"d{seed:04d}-pair{pair:02d}-miss",
                    arrival_ms=base_ms,
                    output_tokens=12,
                    prompt_tokens=512,
                    speculation_length=5,
                    cache_hit_probability=0.15,
                    token_acceptance_probability=0.78,
                    tbt_slo_ms=12.0,
                ),
                RequestConfig(
                    request_id=f"d{seed:04d}-pair{pair:02d}-hit",
                    arrival_ms=base_ms
                    + 3.0
                    + 0.2
                    * timing_rng.uniform(
                        f"pair-{pair}",
                        0,
                        "within-pair-jitter",
                    ),
                    output_tokens=12,
                    prompt_tokens=512,
                    speculation_length=5,
                    cache_hit_probability=0.9,
                    token_acceptance_probability=0.78,
                    tbt_slo_ms=12.0,
                ),
            )
        )
    burst_count = request_count - len(requests)
    burst = poisson_arrivals(
        count=burst_count,
        mean_interarrival_ms=0.08,
        rng=timing_rng,
        process_id=f"mechanism-decoder-burst-{seed}",
        start_ms=pair_count * 25.0,
    )
    requests.extend(
        RequestConfig(
            request_id=f"d{seed:04d}-burst{ordinal:04d}",
            arrival_ms=arrival_ms,
            output_tokens=24,
            prompt_tokens=512,
            speculation_length=5,
            cache_hit_probability=0.72,
            token_acceptance_probability=0.78,
            tbt_slo_ms=12.0,
        )
        for ordinal, arrival_ms in enumerate(burst.times_ms)
    )
    return Workload(
        tuple(requests),
        name=f"mechanism-decoder-composite/seed-{seed:04d}",
    )


def _decoder_metrics(
    setting_id: str,
    setting: DecoderSetting,
    seed: int,
    request_count: int,
) -> tuple[MetricRow, str]:
    workload = _decoder_workload(seed, request_count)
    result = simulate(
        workload,
        _decoder_profile(setting),
        FissionSpecPolicy(max_wait_ms=setting.controller_max_wait_ms),
        CounterRNG(f"mechanism-study/decoder-outcomes/{seed}"),
        max_batch_size=setting.batch_capacity,
    )
    metrics = summarize(result)
    row: MetricRow = {
        "schema_version": SCHEMA_VERSION,
        "evidence_class": "synthetic-cpu-decoder-model",
        "measurement_warning": WARNING,
        "stratum": "decoder-policy",
        "setting_id": setting_id,
        "cluster_id": f"seed-{seed:04d}",
        "seed": seed,
        "requests": metrics.requests,
        "throughput_tokens_per_s": metrics.throughput_tokens_per_s,
        "p95_tbt_ms": metrics.p95_tbt_ms,
        "request_slo_miss_rate": 1.0 - metrics.request_tbt_slo_attainment,
        "target_launches_per_request": metrics.target_launches / metrics.requests,
        "mean_batch": metrics.mean_batch,
        "draft_launches_per_request": metrics.draft_launches / metrics.requests,
    }
    trace_payload = {
        "stratum": "decoder-policy",
        "setting_id": setting_id,
        "seed": seed,
        "setting": asdict(setting),
        "workload": workload,
        "result": result,
    }
    return row, sha256_document(trace_payload)


def _fidelity_requests(seed: int, request_count: int) -> tuple[FidelityRequest, ...]:
    arrivals = poisson_arrivals(
        count=request_count,
        mean_interarrival_ms=0.08,
        rng=CounterRNG(f"mechanism-study/fidelity-arrivals/{seed}"),
        process_id=f"mechanism-fidelity-{seed}",
    )
    return tuple(
        FidelityRequest(
            request_id=f"f{seed:04d}-r{ordinal:04d}",
            arrival_ms=arrival_ms,
            prompt_tokens=256 + 128 * (ordinal % 5),
            output_tokens=16,
            speculation_length=5,
            class_weights=(("repetitive", 0.6), ("diffuse", 0.4)),
            correlation_key=f"tenant-{ordinal // 4}" if ordinal % 3 != 0 else None,
            priority=2 if ordinal % 5 == 0 else 0,
        )
        for ordinal, arrival_ms in enumerate(arrivals.times_ms)
    )


def _fidelity_classes() -> tuple[OutcomeClass, ...]:
    return (
        OutcomeClass("repetitive", 0.9, (0.72, 0.20, 0.08)),
        OutcomeClass("diffuse", 0.55, (0.40, 0.34, 0.26)),
    )


def _fidelity_config(setting: FidelitySetting) -> FidelityConfig:
    costs = ContextCostModel(
        prefill_base_ms=0.15,
        prefill_per_token_ms=0.0004,
        target_base_ms=0.8,
        target_per_row_ms=0.07,
        target_per_context_token_ms=0.00002,
        target_per_verifier_slot_ms=0.018,
        draft_base_ms=0.25,
        recovery_base_ms=0.65,
        draft_per_row_ms=0.03,
        draft_per_context_token_ms=0.00001,
        draft_per_branch_ms=0.025,
        network_base_ms=0.04,
        network_per_byte_ms=0.0000005,
        network_jitter_ms=setting.network_jitter_ms,
    )
    return FidelityConfig(
        costs=costs,
        remote=RemoteDraftConfig(
            workers=2,
            queue_policy="continuous-batching",
            max_batch_size=16,
            batch_window_ms=0.08,
            queue_capacity=8,
            failure_probability=setting.network_failure_probability,
            max_retries=1,
            retry_backoff_ms=0.1,
        ),
        cache_byte_budget=setting.cache_byte_budget,
        cache_page_size_bytes=16 * 1024,
        kv_bytes_per_token=4096,
        continuation_tokens=4,
        fanout=setting.fanout,
        target_batch_size=16,
    )


def _fidelity_metric_values(trace: FidelityTrace) -> dict[str, float]:
    target_end_by_request = {
        request_id: batch.end_ms
        for batch in trace.target_batches
        for request_id in batch.request_ids
    }
    nonterminal = tuple(request for request in trace.requests if not request.terminal)
    if not nonterminal:
        raise MechanismStudyError("fidelity workload unexpectedly has no nonterminal requests")
    lookups = tuple(request for request in nonterminal if request.cache_hit is not None)
    hits = sum(request.cache_hit is True for request in lookups)
    restricted_delays: list[float] = []
    ready_within_5ms = 0
    unready = 0
    for request in nonterminal:
        target_end = target_end_by_request[request.request_id]
        if request.next_ready_ms is None:
            delay = READINESS_RESTRICTION_MS
            unready += 1
        else:
            delay = max(0.0, request.next_ready_ms - target_end)
            if delay <= 5.0:
                ready_within_5ms += 1
        restricted_delays.append(min(delay, READINESS_RESTRICTION_MS))
    attempts = len(trace.precompute_trace.attempts)
    if trace.recovery_trace is not None:
        attempts += len(trace.recovery_trace.attempts)
    return {
        "cache_hit_rate": hits / len(lookups),
        "restricted_next_ready_delay_ms": math.fsum(restricted_delays) / len(nonterminal),
        "next_round_unready_rate": unready / len(nonterminal),
        "mean_ttft_ms": math.fsum(request.ttft_ms for request in trace.requests)
        / len(trace.requests),
        "ready_within_5ms_rate": ready_within_5ms / len(nonterminal),
        "remote_attempts_per_request": attempts / len(trace.requests),
        "cache_evictions_per_request": trace.cache_evictions / len(trace.requests),
    }


def _fidelity_metrics(
    setting_id: str,
    setting: FidelitySetting,
    seed: int,
    request_count: int,
) -> tuple[MetricRow, str]:
    requests = _fidelity_requests(seed, request_count)
    config = _fidelity_config(setting)
    trace = simulate_fidelity_trace(
        requests,
        _fidelity_classes(),
        config,
        seed=f"mechanism-study/fidelity-outcomes/{seed}",
    )
    row: MetricRow = {
        "schema_version": SCHEMA_VERSION,
        "evidence_class": "synthetic-cpu-one-round-fidelity-model",
        "measurement_warning": WARNING,
        "stratum": "one-round-fidelity",
        "setting_id": setting_id,
        "cluster_id": f"seed-{seed:04d}",
        "seed": seed,
        "requests": len(trace.requests),
        **_fidelity_metric_values(trace),
    }
    trace_payload = {
        "stratum": "one-round-fidelity",
        "setting_id": setting_id,
        "seed": seed,
        "setting": asdict(setting),
        "requests": requests,
        "classes": _fidelity_classes(),
        "config": config,
        "trace": trace,
    }
    return row, sha256_document(trace_payload)


def generate_rows(config: ModeConfig) -> list[MetricRow]:
    """Run every paired reference/intervention cluster."""

    rows: list[MetricRow] = []
    decoder_settings = [("decoder/reference", decoder_reference())]
    decoder_settings.extend(
        (intervention.setting_id, _apply_decoder(intervention))
        for intervention in decoder_interventions()
    )
    fidelity_settings = [("fidelity/reference", fidelity_reference())]
    fidelity_settings.extend(
        (intervention.setting_id, _apply_fidelity(intervention))
        for intervention in fidelity_interventions()
    )
    for seed in config.seeds:
        for setting_id, decoder_setting in decoder_settings:
            row, trace_hash = _decoder_metrics(
                setting_id,
                decoder_setting,
                seed,
                config.decoder_requests,
            )
            row["trace_payload_sha256"] = trace_hash
            rows.append(row)
        for setting_id, fidelity_setting in fidelity_settings:
            row, trace_hash = _fidelity_metrics(
                setting_id,
                fidelity_setting,
                seed,
                config.fidelity_requests,
            )
            row["trace_payload_sha256"] = trace_hash
            rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            str(row["stratum"]),
            str(row["setting_id"]),
            int(row["seed"]),
        ),
    )


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise MechanismStudyError("percentile needs at least one value")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + weight * (ordered[upper] - ordered[lower])


def _bootstrap_seed(hypothesis_id: str) -> int:
    digest = hashlib.sha256(f"fissionspec/mechanism-bootstrap/v1/{hypothesis_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _paired_bootstrap_interval(
    differences: tuple[float, ...],
    *,
    confidence_level: float,
    resamples: int,
    hypothesis_id: str,
) -> tuple[float, float]:
    if len(differences) < 2:
        raise MechanismStudyError("paired bootstrap needs at least two clusters")
    rng = random.Random(_bootstrap_seed(hypothesis_id))
    count = len(differences)
    estimates = [
        math.fsum(differences[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(resamples)
    ]
    tail = (1.0 - confidence_level) / 2.0
    return _percentile(estimates, tail), _percentile(estimates, 1.0 - tail)


def _exact_sign_pvalue(differences: tuple[float, ...]) -> float:
    positive = sum(value > 0.0 for value in differences)
    negative = sum(value < 0.0 for value in differences)
    trials = positive + negative
    if trials == 0:
        return 1.0
    lower_tail = sum(math.comb(trials, value) for value in range(min(positive, negative) + 1))
    return min(1.0, 2.0 * lower_tail / float(2**trials))


def _holm_adjust(pvalues: list[float]) -> list[float]:
    indexed = sorted(enumerate(pvalues), key=lambda item: (item[1], item[0]))
    adjusted = [1.0] * len(pvalues)
    running = 0.0
    hypotheses = len(pvalues)
    for rank, (original_index, pvalue) in enumerate(indexed):
        running = max(running, min(1.0, (hypotheses - rank) * pvalue))
        adjusted[original_index] = running
    return adjusted


def analyze_rows(rows: list[MetricRow], config: ModeConfig) -> JsonObject:
    """Compute paired effects with one predeclared simultaneous family."""

    by_key = {(str(row["stratum"]), str(row["setting_id"]), int(row["seed"])): row for row in rows}
    if len(by_key) != len(rows):
        raise MechanismStudyError("duplicate stratum/setting/seed row")
    hypothesis_specs: list[tuple[Intervention, str]] = []
    for intervention in all_interventions():
        metrics = (
            DECODER_CONFIRMATORY_METRICS
            if intervention.stratum == "decoder-policy"
            else FIDELITY_CONFIRMATORY_METRICS
        )
        hypothesis_specs.extend((intervention, metric) for metric in metrics)
    family_size = len(hypothesis_specs)
    per_hypothesis_alpha = FAMILYWISE_ALPHA / family_size
    confidence_level = 1.0 - per_hypothesis_alpha
    comparisons: list[JsonObject] = []
    raw_sign_pvalues: list[float] = []
    for intervention, metric in hypothesis_specs:
        reference_id = (
            "decoder/reference"
            if intervention.stratum == "decoder-policy"
            else "fidelity/reference"
        )
        intervention_values: list[float] = []
        reference_values: list[float] = []
        for seed in config.seeds:
            try:
                intervention_row = by_key[(intervention.stratum, intervention.setting_id, seed)]
                reference_row = by_key[(intervention.stratum, reference_id, seed)]
            except KeyError as exc:
                raise MechanismStudyError(f"unpaired row for {intervention.setting_id}") from exc
            intervention_values.append(float(intervention_row[metric]))
            reference_values.append(float(reference_row[metric]))
        differences = tuple(
            intervention_value - reference_value
            for intervention_value, reference_value in zip(
                intervention_values,
                reference_values,
                strict=True,
            )
        )
        hypothesis_id = f"{intervention.setting_id}/{metric}"
        lower, upper = _paired_bootstrap_interval(
            differences,
            confidence_level=confidence_level,
            resamples=config.bootstrap_resamples,
            hypothesis_id=hypothesis_id,
        )
        sign_pvalue = _exact_sign_pvalue(differences)
        raw_sign_pvalues.append(sign_pvalue)
        comparisons.append(
            {
                "hypothesis_id": hypothesis_id,
                "stratum": intervention.stratum,
                "factor": intervention.factor,
                "level": intervention.level,
                "changed_field": intervention.changed_field,
                "reference_value": intervention.reference_value,
                "intervention_value": intervention.intervention_value,
                "metric": metric,
                "estimand": (
                    "equally weighted mean of paired cluster differences "
                    "(intervention minus shared reference)"
                ),
                "clusters": len(differences),
                "reference_mean": math.fsum(reference_values) / len(reference_values),
                "intervention_mean": math.fsum(intervention_values) / len(intervention_values),
                "mean_difference": math.fsum(differences) / len(differences),
                "simultaneous_interval": {
                    "lower": lower,
                    "upper": upper,
                    "confidence_level": confidence_level,
                    "method": "paired-cluster percentile bootstrap with Bonferroni allocation",
                },
                "positive_clusters": sum(value > 0.0 for value in differences),
                "negative_clusters": sum(value < 0.0 for value in differences),
                "tied_clusters": sum(value == 0.0 for value in differences),
                "exact_two_sided_sign_pvalue": sign_pvalue,
            }
        )
    adjusted = _holm_adjust(raw_sign_pvalues)
    for comparison, pvalue in zip(comparisons, adjusted, strict=True):
        comparison["holm_adjusted_sign_pvalue"] = pvalue
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_class": "synthetic-cpu-model",
        "measurement_warning": WARNING,
        "claim_boundary": CLAIM_BOUNDARY,
        "family": {
            "family_id": "mechanism-study-confirmatory-v1",
            "familywise_alpha": FAMILYWISE_ALPHA,
            "hypotheses": family_size,
            "per_hypothesis_alpha": per_hypothesis_alpha,
            "simultaneous_per_hypothesis_confidence_level": confidence_level,
            "bootstrap_resamples": config.bootstrap_resamples,
            "bootstrap_tail_order_statistics_expected": (
                config.bootstrap_resamples * per_hypothesis_alpha / 2.0
            ),
            "sign_test_multiplicity": "Holm step-down over the same complete family",
            "interpretation": (
                "Bootstrap coverage is an asymptotic cluster-resampling approximation; "
                "Holm-adjusted sign tests are distribution-free for cluster signs under "
                "independent clusters and the sign-null."
            ),
        },
        "confirmatory_metrics": {
            "decoder-policy": list(DECODER_CONFIRMATORY_METRICS),
            "one-round-fidelity": list(FIDELITY_CONFIRMATORY_METRICS),
        },
        "descriptive_metrics_not_in_family": {
            key: list(value) for key, value in DESCRIPTIVE_METRICS.items()
        },
        "comparisons": comparisons,
    }


def design_document(config: ModeConfig) -> JsonObject:
    """Describe the complete frozen intervention and pairing design."""

    intervention_documents: list[JsonObject] = []
    for intervention in all_interventions():
        setting: DecoderSetting | FidelitySetting
        reference: DecoderSetting | FidelitySetting
        if intervention.stratum == "decoder-policy":
            reference = decoder_reference()
            setting = _apply_decoder(intervention)
        else:
            reference = fidelity_reference()
            setting = _apply_fidelity(intervention)
        reference_values = asdict(reference)
        setting_values = asdict(setting)
        changed = [
            field for field in reference_values if reference_values[field] != setting_values[field]
        ]
        intervention_documents.append(
            {
                **asdict(intervention),
                "reference_setting": reference_values,
                "intervention_setting": setting_values,
                "changed_fields_audit": changed,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "study": "FissionSpec one-factor-at-a-time CPU mechanism study",
        "mode": config.mode,
        "evidence_class": "synthetic-cpu-model",
        "measurement_warning": WARNING,
        "claim_boundary": CLAIM_BOUNDARY,
        "pairing": {
            "cluster_unit": "seed",
            "seeds": list(config.seeds),
            "common_random_numbers": (
                "Counter-addressed draws are invariant to intervention scheduling and "
                "are shared within each stratum/seed pair."
            ),
            "reference_reuse": (
                "Each stratum has one shared reference row per seed; every alternate "
                "is compared only with that same-seed reference."
            ),
        },
        "sample_sizes": {
            "decoder_requests_per_cluster": config.decoder_requests,
            "fidelity_requests_per_cluster": config.fidelity_requests,
            "clusters": len(config.seeds),
        },
        "decoder_workload": {
            "construction": (
                "Up to eight sparse two-request miss/hit recovery probes followed by "
                "one high-load Poisson burst."
            ),
            "purpose": (
                "Sparse probes exercise controller wait/refusion uptake; the burst "
                "exercises batch-capacity and physical-slot interventions in the same "
                "immutable workload."
            ),
            "factor_status": "The workload is identical within every paired contrast.",
        },
        "reference_settings": {
            "decoder-policy": asdict(decoder_reference()),
            "one-round-fidelity": asdict(fidelity_reference()),
        },
        "interventions": intervention_documents,
        "inference": {
            "familywise_alpha": FAMILYWISE_ALPHA,
            "bootstrap_resamples": config.bootstrap_resamples,
            "readiness_restriction_ms": READINESS_RESTRICTION_MS,
            "confirmatory_metrics": {
                "decoder-policy": list(DECODER_CONFIRMATORY_METRICS),
                "one-round-fidelity": list(FIDELITY_CONFIRMATORY_METRICS),
            },
        },
        "configuration_search": False,
        "selection_rule": "All predeclared contrasts are reported regardless of sign or magnitude.",
    }


ROW_COLUMNS: Final = (
    "schema_version",
    "evidence_class",
    "measurement_warning",
    "stratum",
    "setting_id",
    "cluster_id",
    "seed",
    "requests",
    "throughput_tokens_per_s",
    "p95_tbt_ms",
    "request_slo_miss_rate",
    "target_launches_per_request",
    "mean_batch",
    "draft_launches_per_request",
    "cache_hit_rate",
    "restricted_next_ready_delay_ms",
    "next_round_unready_rate",
    "mean_ttft_ms",
    "ready_within_5ms_rate",
    "remote_attempts_per_request",
    "cache_evictions_per_request",
    "trace_payload_sha256",
)


def _write_json(path: Path, document: object) -> None:
    path.write_bytes(
        json.dumps(
            _canonical(document),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _write_rows(path: Path, rows: list[MetricRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=ROW_COLUMNS,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in ROW_COLUMNS})


def _read_rows(path: Path) -> list[MetricRow]:
    rows: list[MetricRow] = []
    numeric = set(ROW_COLUMNS) - {
        "evidence_class",
        "measurement_warning",
        "stratum",
        "setting_id",
        "cluster_id",
        "trace_payload_sha256",
    }
    integer_columns = {"schema_version", "seed", "requests"}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ROW_COLUMNS:
            raise MechanismStudyError("rows.csv header does not match the closed schema")
        for line_number, source in enumerate(reader, start=2):
            row: MetricRow = {}
            for column in ROW_COLUMNS:
                value = source[column]
                if column in numeric:
                    if value == "":
                        continue
                    try:
                        parsed: int | float = (
                            int(value) if column in integer_columns else float(value)
                        )
                    except ValueError as exc:
                        raise MechanismStudyError(
                            f"rows.csv line {line_number}: {column} is not numeric"
                        ) from exc
                    if isinstance(parsed, float) and not math.isfinite(parsed):
                        raise MechanismStudyError(
                            f"rows.csv line {line_number}: {column} is not finite"
                        )
                    row[column] = parsed
                else:
                    row[column] = value
            rows.append(row)
    return rows


def _summary_markdown(analysis: JsonObject) -> str:
    comparisons = cast(list[JsonObject], analysis["comparisons"])
    lines = [
        "# FissionSpec CPU mechanism study",
        "",
        f"> {WARNING}",
        "",
        CLAIM_BOUNDARY,
        "",
        "All entries are intervention-minus-reference paired means. Intervals use the",
        "single predeclared family described in `inference.json`. Mixed, null, and adverse",
        "effects are retained.",
        "",
        "| factor / level | metric | paired mean | simultaneous interval |",
        "|---|---:|---:|---:|",
    ]
    for comparison in comparisons:
        interval = cast(JsonObject, comparison["simultaneous_interval"])
        lines.append(
            "| "
            f"{comparison['factor']} / {comparison['level']} | "
            f"{comparison['metric']} | "
            f"{cast(float, comparison['mean_difference']):.6g} | "
            f"[{cast(float, interval['lower']):.6g}, "
            f"{cast(float, interval['upper']):.6g}] |"
        )
    lines.extend(
        (
            "",
            "The table is mechanistic sensitivity evidence inside frozen CPU models.",
            "It is not a policy leaderboard and is not GPU-performance evidence.",
            "",
        )
    )
    return "\n".join(lines)


def environment_document(mode: StudyMode) -> JsonObject:
    """Capture path-independent runtime and archival interpreter provenance."""

    return {
        "schema_version": SCHEMA_VERSION,
        "measurement_warning": WARNING,
        "observed_runtime": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
        "reproduction_contract": {
            "archival_python_version": "3.12.8",
            "dependency_lock": "requirements/repro.lock",
            "container_definition": "repro/Dockerfile.cpu",
            "launcher": "python",
            "command_template": [
                "python",
                "experiments/run_mechanism_study.py",
                "--mode",
                mode,
                "--output-dir",
                "<OUTPUT_DIR>",
            ],
            "path_policy": (
                "Repository-relative program and contract paths; logical output placeholder; "
                "no host executable or output-directory path is serialized."
            ),
        },
    }


def write_bundle(output_dir: Path, config: ModeConfig, *, repo_root: Path) -> JsonObject:
    """Generate, analyze, and hash a complete bounded study bundle."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = generate_rows(config)
    design = design_document(config)
    inference = analyze_rows(rows, config)
    _write_json(output_dir / "design.json", design)
    _write_rows(output_dir / "rows.csv", rows)
    _write_json(output_dir / "inference.json", inference)
    (output_dir / "SUMMARY.md").write_text(
        _summary_markdown(inference),
        encoding="utf-8",
    )
    _write_json(output_dir / "environment.json", environment_document(config.mode))
    implementation_digest, source_hashes = _implementation_sha256(repo_root)
    files = {}
    for filename in (
        "SUMMARY.md",
        "design.json",
        "environment.json",
        "inference.json",
        "rows.csv",
    ):
        payload = (output_dir / filename).read_bytes()
        files[filename] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    manifest_payload: JsonObject = {
        "schema_version": SCHEMA_VERSION,
        "study": "FissionSpec one-factor-at-a-time CPU mechanism study",
        "mode": config.mode,
        "measurement_warning": WARNING,
        "claim_boundary": CLAIM_BOUNDARY,
        "implementation_sha256": implementation_digest,
        "implementation_files": source_hashes,
        "artifact_files": files,
        "rows": len(rows),
        "trace_hashes": len(rows),
        "trace_retention": (
            "rows.csv retains every per-cluster sufficient statistic and the SHA-256 "
            "of its complete canonical simulator/fidelity trace payload."
        ),
    }
    manifest: JsonObject = {
        **manifest_payload,
        "payload_sha256": sha256_document(manifest_payload),
    }
    _write_json(output_dir / "manifest.json", manifest)
    verify_bundle(
        output_dir,
        expected_mode=config.mode,
        repo_root=repo_root,
    )
    return manifest


def verify_bundle(
    output_dir: Path,
    *,
    expected_mode: StudyMode | None = None,
    repo_root: Path | None = None,
) -> JsonObject:
    """Fail closed on hashes, pairing, schema, and frozen design invariants."""

    manifest_path = output_dir / "manifest.json"
    try:
        manifest = cast(JsonObject, json.loads(manifest_path.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MechanismStudyError("cannot read manifest.json") from exc
    expected_manifest_fields = {
        "schema_version",
        "study",
        "mode",
        "measurement_warning",
        "claim_boundary",
        "implementation_sha256",
        "implementation_files",
        "artifact_files",
        "rows",
        "trace_hashes",
        "trace_retention",
        "payload_sha256",
    }
    if set(manifest) != expected_manifest_fields:
        raise MechanismStudyError("manifest has unexpected or missing fields")
    supplied_payload_hash = manifest.get("payload_sha256")
    if (
        not isinstance(supplied_payload_hash, str)
        or len(supplied_payload_hash) != 64
        or any(character not in "0123456789abcdef" for character in supplied_payload_hash)
    ):
        raise MechanismStudyError("manifest payload SHA-256 is malformed")
    manifest_payload = dict(manifest)
    manifest_payload.pop("payload_sha256")
    if sha256_document(manifest_payload) != supplied_payload_hash:
        raise MechanismStudyError("manifest payload hash mismatch")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise MechanismStudyError("manifest schema version mismatch")
    if (
        manifest.get("study") != "FissionSpec one-factor-at-a-time CPU mechanism study"
        or manifest.get("measurement_warning") != WARNING
        or manifest.get("claim_boundary") != CLAIM_BOUNDARY
    ):
        raise MechanismStudyError("manifest evidence boundary drifted")
    mode = manifest.get("mode")
    if mode not in {"ci", "full"}:
        raise MechanismStudyError("manifest mode is invalid")
    if expected_mode is not None and mode != expected_mode:
        raise MechanismStudyError(f"expected mode {expected_mode!r}, found {mode!r}")
    files = manifest.get("artifact_files")
    if not isinstance(files, dict):
        raise MechanismStudyError("manifest artifact_files must be an object")
    if set(files) != {
        "SUMMARY.md",
        "design.json",
        "environment.json",
        "inference.json",
        "rows.csv",
    }:
        raise MechanismStudyError("manifest artifact file set is not frozen")
    expected_entries = set(files) | {"manifest.json"}
    actual_entries = {path.name for path in output_dir.iterdir()}
    if actual_entries != expected_entries:
        raise MechanismStudyError(
            "bundle entries do not match the closed manifest: "
            f"unexpected={sorted(actual_entries - expected_entries)}, "
            f"missing={sorted(expected_entries - actual_entries)}"
        )
    for filename, record in files.items():
        if not isinstance(filename, str) or not isinstance(record, dict):
            raise MechanismStudyError("malformed artifact file record")
        if set(record) != {"bytes", "sha256"}:
            raise MechanismStudyError("artifact file record has unexpected fields")
        path = output_dir / filename
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != record.get("sha256"):
            raise MechanismStudyError(f"artifact hash mismatch: {filename}")
        if len(payload) != record.get("bytes"):
            raise MechanismStudyError(f"artifact byte count mismatch: {filename}")
    if repo_root is not None:
        implementation_digest, source_hashes = _implementation_sha256(repo_root)
        if source_hashes != manifest.get("implementation_files"):
            raise MechanismStudyError("implementation source hashes do not match the bundle")
        if implementation_digest != manifest.get("implementation_sha256"):
            raise MechanismStudyError("aggregate implementation hash does not match the bundle")
    design = cast(
        JsonObject,
        json.loads((output_dir / "design.json").read_text(encoding="utf-8")),
    )
    interventions = design.get("interventions")
    if not isinstance(interventions, list) or len(interventions) != len(all_interventions()):
        raise MechanismStudyError("design intervention set is incomplete")
    for intervention in interventions:
        if not isinstance(intervention, dict):
            raise MechanismStudyError("malformed intervention")
        if intervention.get("changed_fields_audit") != [intervention.get("changed_field")]:
            raise MechanismStudyError("an intervention is not one-factor-at-a-time")
    rows = _read_rows(output_dir / "rows.csv")
    seeds = mode_config(cast(StudyMode, mode)).seeds
    expected_rows = len(seeds) * (2 + len(decoder_interventions()) + len(fidelity_interventions()))
    if len(rows) != expected_rows or len(rows) != manifest.get("rows"):
        raise MechanismStudyError("row count does not match the frozen design")
    if manifest.get("trace_hashes") != len(rows):
        raise MechanismStudyError("trace hash count does not match the frozen design")
    for row in rows:
        trace_hash = str(row["trace_payload_sha256"])
        if len(trace_hash) != 64 or any(
            character not in "0123456789abcdef" for character in trace_hash
        ):
            raise MechanismStudyError("invalid trace payload SHA-256")
        if row["measurement_warning"] != WARNING:
            raise MechanismStudyError("row is missing the CPU-only warning")
    inference = cast(
        JsonObject,
        json.loads((output_dir / "inference.json").read_text(encoding="utf-8")),
    )
    comparisons = inference.get("comparisons")
    family = inference.get("family")
    if not isinstance(comparisons, list) or not isinstance(family, dict):
        raise MechanismStudyError("malformed inference artifact")
    if len(comparisons) != family.get("hypotheses") or len(comparisons) != 48:
        raise MechanismStudyError("confirmatory family size mismatch")
    expected_pairs = {
        (intervention.stratum, intervention.setting_id, seed)
        for intervention in all_interventions()
        for seed in seeds
    }
    available = {(str(row["stratum"]), str(row["setting_id"]), int(row["seed"])) for row in rows}
    if not expected_pairs <= available:
        raise MechanismStudyError("one or more intervention clusters are absent")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("ci", "full"), default="ci")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/mechanism_study"),
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify an existing bundle instead of regenerating it",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.verify:
            manifest = verify_bundle(
                args.output_dir,
                expected_mode=args.mode,
                repo_root=Path(__file__).resolve().parents[1],
            )
        else:
            repo_root = Path(__file__).resolve().parents[1]
            manifest = write_bundle(
                args.output_dir,
                mode_config(args.mode),
                repo_root=repo_root,
            )
    except (MechanismStudyError, OSError, ValueError) as exc:
        print(f"mechanism study failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
