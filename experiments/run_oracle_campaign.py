#!/usr/bin/env python3
"""Run the expanded exact CPU scheduler-oracle campaign.

This campaign compiles pre-realized hit, miss, retry, cancellation, and
terminal-failure scenarios into the one-shot finite scheduling domain of
``fissionspec.general_oracle``.  Every problem is solved exactly by dynamic
programming; a stratified subset is independently re-enumerated without
memoization.  All built-in Python dispatch policies and the independent EDF
baseline are evaluated over the same fixed-admission domain.

The evidence is deliberately bounded: at most three non-preemptive target jobs
and one launch per active job.  It is CPU proof/model evidence, not a GPU
measurement and not a full multi-round serving optimum.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import os
import platform
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Final, Literal, Protocol, TypeAlias, cast

from fissionspec.general_oracle import (
    DispatchEvent,
    ExactLatencySurface,
    GeneralOracleCertificate,
    OracleCapacity,
    OracleJob,
    OracleObjective,
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
from fissionspec.policies import (
    DispatchContext,
    FissionSpecPolicy,
    FixedCoalescePolicy,
    ImmediateFissionPolicy,
    SaguaroBarrierPolicy,
    SchedulingPolicy,
    SPECTREPaddedPolicy,
)
from fissionspec.profiles import HardwareProfile

SCHEMA_VERSION: Final = 1
WARNING: Final = "EXACT BOUNDED CPU MODEL / NOT A GPU MEASUREMENT"
CLAIM_BOUNDARY: Final = (
    "Exactness holds only for the declared one-shot, non-preemptive, at-most-three-job "
    "finite action space. It does not establish a multi-round serving optimum or "
    "accelerator performance."
)
StudyMode = Literal["ci", "full"]
PhysicalMode = Literal["packed-slots", "graph-bucket-slots"]
CapacityMode = Literal["tight", "wide"]
OutcomePattern = Literal[
    "head-miss",
    "tail-miss",
    "double-miss",
    "retry-once",
    "canceled-tail",
    "terminal-failure-middle",
]
ScenarioFamily = Literal["main", "retry", "cancellation", "terminal-failure"]
MetricValue: TypeAlias = str | int | float
ComparisonRow: TypeAlias = dict[str, MetricValue]
JsonObject: TypeAlias = dict[str, object]

ARRIVAL_PATTERNS: Final[dict[str, tuple[Fraction, Fraction, Fraction]]] = {
    "synchronized": (Fraction(0), Fraction(0), Fraction(0)),
    "staggered": (Fraction(0), Fraction(1, 2), Fraction(1)),
    "recovery-wave": (Fraction(0), Fraction(1, 4), Fraction(3, 2)),
}
RECOVERY_ETAS: Final = (Fraction(1, 4), Fraction(1))
RECOVERY_JITTERS: Final = (Fraction(0), Fraction(1, 4))
PHYSICAL_MODES: Final[tuple[PhysicalMode, ...]] = (
    "packed-slots",
    "graph-bucket-slots",
)
CAPACITY_MODES: Final[tuple[CapacityMode, ...]] = ("tight", "wide")
MAX_WAITS: Final = (Fraction(0), Fraction(1, 2), Fraction(1))
DEADLINE_GUARDS: Final = (Fraction(0), Fraction(1, 4))
POLICY_NAMES: Final = (
    "work-conserving-edf",
    "saguaro-barrier",
    "spectre-parallel-padded",
    "immediate-fission",
    "fixed-coalesce",
    "fissionspec-horizon-2",
)
IMPLEMENTATION_PATHS: Final = (
    "experiments/run_oracle_campaign.py",
    "src/fissionspec/general_oracle.py",
    "src/fissionspec/policies.py",
    "src/fissionspec/profiles.py",
)


class OracleCampaignError(ValueError):
    """Raised when campaign design, execution, or artifacts are inconsistent."""


class _PolicyProfile(Protocol):
    def target_latency_ms(self, batch_rows: int, verifier_slots: int) -> float: ...


@dataclass(frozen=True, slots=True)
class ModeConfig:
    """Finite campaign budget."""

    mode: StudyMode
    independent_proofs: int


def mode_config(mode: StudyMode) -> ModeConfig:
    if mode == "ci":
        return ModeConfig(mode="ci", independent_proofs=1)
    if mode == "full":
        return ModeConfig(mode="full", independent_proofs=24)
    raise ValueError(f"unknown mode: {mode!r}")


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    """One pre-realized serving scenario before exact-problem compilation."""

    scenario_id: str
    family: ScenarioFamily
    arrival_pattern: str
    outcome_pattern: OutcomePattern
    recovery_eta: Fraction
    recovery_jitter: Fraction
    physical_mode: PhysicalMode
    capacity_mode: CapacityMode


@dataclass(frozen=True, slots=True)
class ControllerSetting:
    """Controller-only factors that do not change the exact problem."""

    setting_id: str
    max_wait: Fraction
    deadline_guard: Fraction


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _fraction_slug(value: Fraction) -> str:
    return f"{value.numerator}of{value.denominator}"


def _scenario_id(
    *,
    family: ScenarioFamily,
    arrival: str,
    outcome: OutcomePattern,
    eta: Fraction,
    jitter: Fraction,
    physical: PhysicalMode,
    capacity: CapacityMode,
) -> str:
    return (
        f"{family}/{arrival}/{outcome}/eta-{_fraction_slug(eta)}/"
        f"jitter-{_fraction_slug(jitter)}/{physical}/{capacity}"
    )


def full_scenario_specs() -> tuple[ScenarioSpec, ...]:
    """Return 216 predeclared, deterministic, distinct exact problems."""

    specs: list[ScenarioSpec] = []
    for (
        arrival,
        outcome,
        eta,
        jitter,
        physical,
        capacity,
    ) in itertools.product(
        ARRIVAL_PATTERNS,
        ("head-miss", "tail-miss", "double-miss"),
        RECOVERY_ETAS,
        RECOVERY_JITTERS,
        PHYSICAL_MODES,
        CAPACITY_MODES,
    ):
        typed_outcome = cast(OutcomePattern, outcome)
        specs.append(
            ScenarioSpec(
                scenario_id=_scenario_id(
                    family="main",
                    arrival=arrival,
                    outcome=typed_outcome,
                    eta=eta,
                    jitter=jitter,
                    physical=physical,
                    capacity=capacity,
                ),
                family="main",
                arrival_pattern=arrival,
                outcome_pattern=typed_outcome,
                recovery_eta=eta,
                recovery_jitter=jitter,
                physical_mode=physical,
                capacity_mode=capacity,
            )
        )
    for arrival, eta, jitter, physical, capacity in itertools.product(
        ARRIVAL_PATTERNS,
        RECOVERY_ETAS,
        RECOVERY_JITTERS,
        PHYSICAL_MODES,
        CAPACITY_MODES,
    ):
        specs.append(
            ScenarioSpec(
                scenario_id=_scenario_id(
                    family="retry",
                    arrival=arrival,
                    outcome="retry-once",
                    eta=eta,
                    jitter=jitter,
                    physical=physical,
                    capacity=capacity,
                ),
                family="retry",
                arrival_pattern=arrival,
                outcome_pattern="retry-once",
                recovery_eta=eta,
                recovery_jitter=jitter,
                physical_mode=physical,
                capacity_mode=capacity,
            )
        )
    for family, outcome in (
        ("cancellation", "canceled-tail"),
        ("terminal-failure", "terminal-failure-middle"),
    ):
        typed_family = cast(ScenarioFamily, family)
        typed_outcome = cast(OutcomePattern, outcome)
        for arrival, physical, capacity in itertools.product(
            ARRIVAL_PATTERNS,
            PHYSICAL_MODES,
            CAPACITY_MODES,
        ):
            specs.append(
                ScenarioSpec(
                    scenario_id=_scenario_id(
                        family=typed_family,
                        arrival=arrival,
                        outcome=typed_outcome,
                        eta=Fraction(0),
                        jitter=Fraction(0),
                        physical=physical,
                        capacity=capacity,
                    ),
                    family=typed_family,
                    arrival_pattern=arrival,
                    outcome_pattern=typed_outcome,
                    recovery_eta=Fraction(0),
                    recovery_jitter=Fraction(0),
                    physical_mode=physical,
                    capacity_mode=capacity,
                )
            )
    ordered = tuple(sorted(specs, key=lambda spec: spec.scenario_id))
    if len(ordered) != 216 or len({spec.scenario_id for spec in ordered}) != 216:
        raise OracleCampaignError("full scenario matrix must contain 216 unique IDs")
    return ordered


def campaign_specs(mode: StudyMode) -> tuple[ScenarioSpec, ...]:
    """Select the complete matrix or a factor-covering CI subset."""

    complete = full_scenario_specs()
    if mode == "full":
        return complete
    if mode != "ci":
        raise ValueError(f"unknown mode: {mode!r}")
    selected: list[ScenarioSpec] = []
    for outcome in (
        "head-miss",
        "tail-miss",
        "double-miss",
        "retry-once",
        "canceled-tail",
        "terminal-failure-middle",
    ):
        selected.append(next(spec for spec in complete if spec.outcome_pattern == outcome))
    selected.append(
        next(
            spec
            for spec in complete
            if spec.physical_mode == "graph-bucket-slots"
            and spec.capacity_mode == "wide"
            and spec.outcome_pattern == "double-miss"
        )
    )
    selected.append(
        next(
            spec
            for spec in complete
            if spec.arrival_pattern == "staggered"
            and spec.recovery_eta == 1
            and spec.recovery_jitter == Fraction(1, 4)
            and spec.outcome_pattern == "retry-once"
        )
    )
    return tuple(
        sorted({spec.scenario_id: spec for spec in selected}.values(), key=lambda x: x.scenario_id)
    )


def controller_settings() -> tuple[ControllerSetting, ...]:
    return tuple(
        ControllerSetting(
            setting_id=(f"wait-{_fraction_slug(max_wait)}/guard-{_fraction_slug(deadline_guard)}"),
            max_wait=max_wait,
            deadline_guard=deadline_guard,
        )
        for max_wait, deadline_guard in itertools.product(MAX_WAITS, DEADLINE_GUARDS)
    )


def _next_power_of_two(value: int) -> int:
    if value <= 0:
        raise ValueError("bucket input must be positive")
    return 1 << (value - 1).bit_length()


def _latency_duration(mode: PhysicalMode, rows: int, slots: int) -> Fraction:
    slot_work = slots if mode == "packed-slots" else _next_power_of_two(slots)
    slot_coefficient = Fraction(1, 10) if mode == "packed-slots" else Fraction(1, 8)
    return Fraction(1, 2) + Fraction(rows, 4) + slot_coefficient * slot_work


def _capacity(mode: CapacityMode) -> OracleCapacity:
    if mode == "tight":
        return OracleCapacity(row_limit=2, slot_limit=4)
    return OracleCapacity(row_limit=3, slot_limit=6)


def _active_jobs(spec: ScenarioSpec) -> tuple[OracleJob, ...]:
    releases = dict(zip(("a", "b", "c"), ARRIVAL_PATTERNS[spec.arrival_pattern], strict=True))
    if spec.outcome_pattern == "head-miss":
        releases["a"] += spec.recovery_eta + spec.recovery_jitter
    elif spec.outcome_pattern == "tail-miss":
        releases["c"] += spec.recovery_eta + spec.recovery_jitter
    elif spec.outcome_pattern == "double-miss":
        releases["a"] += spec.recovery_eta
        releases["c"] += spec.recovery_eta + spec.recovery_jitter
    elif spec.outcome_pattern == "retry-once":
        releases["b"] += 2 * spec.recovery_eta + spec.recovery_jitter
    elif spec.outcome_pattern == "canceled-tail":
        releases.pop("c")
    elif spec.outcome_pattern == "terminal-failure-middle":
        releases.pop("b")
    else:  # pragma: no cover - closed literal guard
        raise OracleCampaignError(f"unsupported outcome {spec.outcome_pattern!r}")
    widths = {"a": 1, "b": 2, "c": 3}
    deadlines = {"a": Fraction(9, 4), "b": Fraction(15, 4), "c": Fraction(9, 2)}
    weights = {"a": Fraction(3), "b": Fraction(1), "c": Fraction(2)}
    jobs = tuple(
        OracleJob(
            job_id=job_id,
            release_time=release,
            width=widths[job_id],
            deadline=deadlines[job_id],
            weight=weights[job_id],
            cohort_id=f"ready-{_fraction_slug(release)}",
        )
        for job_id, release in sorted(releases.items())
    )
    for job in jobs:
        if job.release_time > job.deadline:
            raise OracleCampaignError(f"{spec.scenario_id} releases {job.job_id} after deadline")
    return jobs


def _admissible_subsets(
    jobs: tuple[OracleJob, ...],
    capacity: OracleCapacity,
) -> tuple[tuple[OracleJob, ...], ...]:
    subsets: list[tuple[OracleJob, ...]] = []
    for size in range(1, min(len(jobs), capacity.row_limit) + 1):
        for selected in itertools.combinations(jobs, size):
            if sum(job.width for job in selected) <= capacity.slot_limit:
                subsets.append(selected)
    return tuple(subsets)


def compile_problem(spec: ScenarioSpec) -> OracleProblem:
    """Compile serving metadata into the exact one-shot scheduling domain."""

    jobs = _active_jobs(spec)
    capacity = _capacity(spec.capacity_mode)
    subsets = _admissible_subsets(jobs, capacity)
    entries = {
        (len(selected), sum(job.width for job in selected)): _latency_duration(
            spec.physical_mode,
            len(selected),
            sum(job.width for job in selected),
        )
        for selected in subsets
    }
    grid: set[Fraction] = set()
    for job in jobs:
        for max_wait in MAX_WAITS:
            grid.add(job.release_time + max_wait)
    for selected in subsets:
        duration = entries[(len(selected), sum(job.width for job in selected))]
        for job in selected:
            for guard in DEADLINE_GUARDS:
                safe = job.deadline - guard - duration
                if safe >= 0:
                    grid.add(safe)
    return OracleProblem(
        jobs,
        capacity,
        ExactLatencySurface(entries),
        wait=OracleWaitConfig(
            include_release_times=True,
            include_deadline_safe_times=True,
            grid_times=tuple(sorted(grid)),
        ),
    )


class ExactPolicyProfile:
    """Float adapter over the campaign's exact latency equation."""

    __slots__ = ("mode",)

    def __init__(self, mode: PhysicalMode) -> None:
        self.mode = mode

    def target_latency_ms(self, batch_rows: int, verifier_slots: int) -> float:
        return float(_latency_duration(self.mode, batch_rows, verifier_slots))


def _selected_ready(
    problem: OracleProblem,
    remaining: set[int],
    now: Fraction,
) -> tuple[int, ...]:
    ready = sorted(
        (index for index in remaining if problem.jobs[index].release_time <= now),
        key=lambda index: (
            problem.jobs[index].deadline,
            problem.jobs[index].job_id,
        ),
    )
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
    return tuple(selected)


def _future_admission(
    problem: OracleProblem,
    remaining: set[int],
    now: Fraction,
) -> tuple[Fraction | None, tuple[int, ...]]:
    release = min(
        (
            problem.jobs[index].release_time
            for index in remaining
            if problem.jobs[index].release_time > now
        ),
        default=None,
    )
    if release is None:
        return None, ()
    cohort = {index for index in remaining if problem.jobs[index].release_time == release}
    return release, _selected_ready(problem, cohort, release)


def _match_wait_point(
    problem: OracleProblem,
    now: Fraction,
    dispatch_at: float,
) -> tuple[Fraction, WaitKind]:
    matches = [
        (time, kind)
        for time, kind in problem.decision_points
        if time > now and math.isclose(dispatch_at, float(time), rel_tol=0.0, abs_tol=1e-12)
    ]
    if not matches:
        raise OracleCampaignError(
            f"policy returned wait {dispatch_at} outside the oracle action space at {now}"
        )
    return min(matches, key=lambda item: item[0])


def evaluate_dispatch_policy(
    problem: OracleProblem,
    spec: ScenarioSpec,
    setting: ControllerSetting,
    policy: SchedulingPolicy,
) -> ScheduleEvaluation:
    """Evaluate one built-in policy with fixed EDF/slot admission."""

    remaining = set(range(len(problem.jobs)))
    now = problem.start_time
    events: list[DispatchEvent | WaitEvent] = []
    completions: dict[str, Fraction] = {}
    profile = cast(HardwareProfile, ExactPolicyProfile(spec.physical_mode))
    iterations = 0
    while remaining:
        iterations += 1
        if iterations > 1_000:
            raise OracleCampaignError("dispatch-policy adapter did not make progress")
        selected = _selected_ready(problem, remaining, now)
        if not selected:
            wake = min(problem.jobs[index].release_time for index in remaining)
            events.append(WaitEvent(now, wake, WaitKind.FORCED_RELEASE))
            now = wake
            continue
        future_release, future_indices = _future_admission(problem, remaining, now)
        selected_slots = sum(problem.jobs[index].width for index in selected)
        future_slots = sum(problem.jobs[index].width for index in future_indices)
        fusion_fits = (
            len(selected) + len(future_indices) <= problem.capacity.row_limit
            and selected_slots + future_slots <= problem.capacity.slot_limit
        )
        context_capacity = problem.capacity.row_limit if fusion_fits else len(selected)
        guarded_deadlines = tuple(
            float(problem.jobs[index].deadline - setting.deadline_guard) for index in selected
        )
        future_guarded_deadlines = tuple(
            float(problem.jobs[index].deadline - setting.deadline_guard) for index in future_indices
        )
        context = DispatchContext(
            now_ms=float(now),
            ready_count=len(selected),
            capacity=context_capacity,
            oldest_ready_ms=min(float(problem.jobs[index].release_time) for index in selected),
            earliest_deadline_ms=guarded_deadlines[0],
            row_slots=tuple(problem.jobs[index].width for index in selected),
            row_deadlines_ms=guarded_deadlines,
            profile=profile,
            next_ready_time_ms=(float(future_release) if future_release is not None else None),
            next_ready_count=len(future_indices),
            earliest_future_deadline_ms=(
                future_guarded_deadlines[0] if future_guarded_deadlines else None
            ),
            future_row_slots=tuple(problem.jobs[index].width for index in future_indices),
            future_row_deadlines_ms=future_guarded_deadlines,
        )
        dispatch_float = policy.dispatch_at(context)
        if not math.isfinite(dispatch_float) or dispatch_float < float(now) - 1e-12:
            raise OracleCampaignError("policy returned a non-finite or past dispatch")
        if dispatch_float > float(now) + 1e-12:
            wait_until, wait_kind = _match_wait_point(
                problem,
                now,
                dispatch_float,
            )
            events.append(WaitEvent(now, wait_until, wait_kind))
            now = wait_until
            continue
        duration = problem.latency.duration(len(selected), selected_slots)
        end = now + duration
        job_ids = tuple(problem.jobs[index].job_id for index in selected)
        events.append(
            DispatchEvent(
                start_time=now,
                end_time=end,
                job_ids=job_ids,
                rows=len(selected),
                slots=selected_slots,
            )
        )
        for index in selected:
            completions[problem.jobs[index].job_id] = end
        remaining.difference_update(selected)
        now = end
    objective = score_completion_times(problem, completions)
    return ScheduleEvaluation(
        input_hash=problem.input_hash,
        events=tuple(events),
        objective=objective,
        completion_times=tuple((job.job_id, completions[job.job_id]) for job in problem.jobs),
    )


def _policies(setting: ControllerSetting) -> tuple[SchedulingPolicy, ...]:
    return (
        SaguaroBarrierPolicy(),
        SPECTREPaddedPolicy(),
        ImmediateFissionPolicy(),
        FixedCoalescePolicy(coalesce_ms=float(setting.max_wait)),
        FissionSpecPolicy(max_wait_ms=float(setting.max_wait)),
    )


def _policy_scope(policy_name: str) -> str:
    if policy_name in {"saguaro-barrier", "spectre-parallel-padded"}:
        return (
            "dispatch-timing component only; barrier recovery and padded target "
            "execution are pre-realized outside this one-shot oracle"
        )
    if policy_name == "work-conserving-edf":
        return "independent fixed-admission no-wait baseline"
    return "complete built-in dispatch_at timing policy under fixed EDF/slot admission"


def _limits() -> OracleSearchLimits:
    return OracleSearchLimits(
        max_jobs=3,
        max_states=200_000,
        max_transitions=2_000_000,
        max_trace_events=128,
    )


def _proof_ids(
    specs: tuple[ScenarioSpec, ...],
    count: int,
) -> frozenset[str]:
    available = {spec.scenario_id: spec for spec in specs}
    selected: list[str] = []

    # Exhaustive enumeration grows sharply for three jobs under packed physical
    # widths. Preserve every outcome and both capacity levels in the independent
    # three-job proofs using graph buckets, then cover both physical modes with
    # the two-job cancellation/failure strata.
    for outcome, physical, capacity, arrival in itertools.product(
        ("canceled-tail", "terminal-failure-middle"),
        PHYSICAL_MODES,
        CAPACITY_MODES,
        ("recovery-wave", "synchronized"),
    ):
        match = next(
            (
                spec
                for spec in specs
                if spec.outcome_pattern == outcome
                and spec.physical_mode == physical
                and spec.capacity_mode == capacity
                and spec.arrival_pattern == arrival
            ),
            None,
        )
        if match is not None:
            selected.append(match.scenario_id)
    for outcome, capacity in itertools.product(
        ("head-miss", "tail-miss", "double-miss", "retry-once"),
        CAPACITY_MODES,
    ):
        match = next(
            (
                spec
                for spec in specs
                if spec.outcome_pattern == outcome
                and spec.physical_mode == "graph-bucket-slots"
                and spec.capacity_mode == capacity
                and spec.arrival_pattern == "recovery-wave"
                and spec.recovery_eta == 1
                and spec.recovery_jitter == 0
            ),
            None,
        )
        if match is not None:
            selected.append(match.scenario_id)
    selected.extend(spec.scenario_id for spec in specs if spec.scenario_id not in set(selected))
    selected = list(dict.fromkeys(selected))
    if any(scenario_id not in available for scenario_id in selected):
        raise OracleCampaignError("independent-proof selection escaped the scenario matrix")
    return frozenset(selected[:count])


def _jsonable(value: object) -> object:
    if isinstance(value, Fraction):
        return _fraction_text(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"cannot serialize {type(value).__name__}")


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def sha256_document(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _objective_fields(prefix: str, objective: OracleObjective) -> dict[str, int | str]:
    return {
        f"{prefix}_deadline_violations": objective.deadline_violations,
        f"{prefix}_weighted_flow": _fraction_text(objective.weighted_flow),
    }


def _comparison_row(
    *,
    spec: ScenarioSpec,
    setting: ControllerSetting,
    policy_name: str,
    candidate: ScheduleEvaluation,
    optimum: GeneralOracleCertificate,
) -> ComparisonRow:
    gap = objective_gap(candidate.objective, optimum.objective)
    if gap.deadline_violation_gap < 0 or (
        gap.deadline_violation_gap == 0 and gap.weighted_flow_gap < 0
    ):
        raise OracleCampaignError("candidate beat the declared exact optimum")
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_class": "exact-bounded-cpu-model",
        "measurement_warning": WARNING,
        "scenario_id": spec.scenario_id,
        "family": spec.family,
        "arrival_pattern": spec.arrival_pattern,
        "outcome_pattern": spec.outcome_pattern,
        "recovery_eta": _fraction_text(spec.recovery_eta),
        "recovery_jitter": _fraction_text(spec.recovery_jitter),
        "physical_mode": spec.physical_mode,
        "capacity_mode": spec.capacity_mode,
        "controller_setting": setting.setting_id,
        "max_wait": _fraction_text(setting.max_wait),
        "deadline_guard": _fraction_text(setting.deadline_guard),
        "policy": policy_name,
        "comparison_scope": _policy_scope(policy_name),
        "problem_hash": optimum.input_hash,
        "certificate_hash": optimum.certificate_hash,
        **_objective_fields("oracle", optimum.objective),
        **_objective_fields("candidate", candidate.objective),
        "deadline_violation_gap": gap.deadline_violation_gap,
        "weighted_flow_gap": _fraction_text(gap.weighted_flow_gap),
        "lexicographically_optimal": int(candidate.objective == optimum.objective),
        "candidate_trace_sha256": sha256_document(candidate.events),
    }


def _nearest_rank(values: Sequence[Fraction], quantile: Fraction) -> Fraction:
    if not values:
        return Fraction()
    ordered = sorted(values)
    rank = max(1, math.ceil(float(quantile) * len(ordered)))
    return ordered[rank - 1]


def _parse_fraction(value: MetricValue) -> Fraction:
    return Fraction(str(value))


def _policy_summaries(rows: list[ComparisonRow]) -> list[JsonObject]:
    grouped: dict[str, list[ComparisonRow]] = defaultdict(list)
    for row in rows:
        grouped[str(row["policy"])].append(row)
    summaries: list[JsonObject] = []
    for policy_name in POLICY_NAMES:
        policy_rows = grouped[policy_name]
        same_deadline_flow = [
            _parse_fraction(row["weighted_flow_gap"])
            for row in policy_rows
            if int(row["deadline_violation_gap"]) == 0
        ]
        deadline_gaps = [int(row["deadline_violation_gap"]) for row in policy_rows]
        summaries.append(
            {
                "policy": policy_name,
                "comparison_scope": _policy_scope(policy_name),
                "cases": len(policy_rows),
                "lexicographically_optimal_cases": sum(
                    int(row["lexicographically_optimal"]) for row in policy_rows
                ),
                "counterexample_cases": sum(
                    not int(row["lexicographically_optimal"]) for row in policy_rows
                ),
                "deadline_regret_cases": sum(gap > 0 for gap in deadline_gaps),
                "maximum_deadline_violation_gap": max(deadline_gaps, default=0),
                "same-deadline-flow-regret": {
                    "cases": len(same_deadline_flow),
                    "p50": _fraction_text(_nearest_rank(same_deadline_flow, Fraction(1, 2))),
                    "p95": _fraction_text(_nearest_rank(same_deadline_flow, Fraction(19, 20))),
                    "maximum": _fraction_text(max(same_deadline_flow, default=Fraction())),
                },
            }
        )
    return summaries


def _counterexample_documents(
    rows: list[ComparisonRow],
    schedule_by_key: Mapping[tuple[str, str, str], ScheduleEvaluation],
    certificate_by_id: Mapping[str, GeneralOracleCertificate],
) -> list[JsonObject]:
    grouped: dict[str, list[ComparisonRow]] = defaultdict(list)
    for row in rows:
        if not int(row["lexicographically_optimal"]):
            grouped[str(row["policy"])].append(row)
    output: list[JsonObject] = []
    for policy in POLICY_NAMES:
        candidates = grouped[policy]
        if not candidates:
            continue
        minimal = min(
            candidates,
            key=lambda row: (
                str(row["scenario_id"]),
                str(row["controller_setting"]),
            ),
        )
        worst = max(
            candidates,
            key=lambda row: (
                int(row["deadline_violation_gap"]),
                (
                    _parse_fraction(row["weighted_flow_gap"])
                    if int(row["deadline_violation_gap"]) == 0
                    else Fraction()
                ),
                str(row["scenario_id"]),
            ),
        )
        for label, row in (("minimal-canonical", minimal), ("worst-lexicographic", worst)):
            key = (
                str(row["scenario_id"]),
                str(row["controller_setting"]),
                policy,
            )
            candidate = schedule_by_key[key]
            certificate = certificate_by_id[str(row["scenario_id"])]
            output.append(
                {
                    "policy": policy,
                    "selection": label,
                    "scenario_id": row["scenario_id"],
                    "controller_setting": row["controller_setting"],
                    "deadline_violation_gap": row["deadline_violation_gap"],
                    "weighted_flow_gap": row["weighted_flow_gap"],
                    "oracle_objective": certificate.objective,
                    "candidate_objective": candidate.objective,
                    "oracle_events": certificate.events,
                    "candidate_events": candidate.events,
                    "problem_hash": certificate.input_hash,
                    "certificate_hash": certificate.certificate_hash,
                }
            )
    return output


def _shift_problem(problem: OracleProblem, delta: Fraction) -> OracleProblem:
    latest = problem.wait.latest_optional_time
    return OracleProblem(
        tuple(
            OracleJob(
                job_id=job.job_id,
                release_time=job.release_time + delta,
                width=job.width,
                deadline=job.deadline + delta,
                weight=job.weight,
                cohort_id=job.cohort_id,
            )
            for job in problem.jobs
        ),
        problem.capacity,
        ExactLatencySurface(
            {(rows, slots): duration for rows, slots, duration in problem.latency.entries}
        ),
        wait=OracleWaitConfig(
            include_release_times=problem.wait.include_release_times,
            include_deadline_safe_times=problem.wait.include_deadline_safe_times,
            grid_times=tuple(time + delta for time in problem.wait.grid_times),
            latest_optional_time=latest + delta if latest is not None else None,
        ),
        start_time=problem.start_time + delta,
    )


def _metamorphic_validations(
    *,
    specs: tuple[ScenarioSpec, ...],
    problem_by_id: Mapping[str, OracleProblem],
    certificate_by_id: Mapping[str, GeneralOracleCertificate],
    proof_ids: frozenset[str],
    rows: list[ComparisonRow],
    schedule_by_key: Mapping[tuple[str, str, str], ScheduleEvaluation],
) -> JsonObject:
    order_invariance = 0
    shift_invariance = 0
    for scenario_id in sorted(proof_ids):
        problem = problem_by_id[scenario_id]
        reversed_problem = OracleProblem(
            tuple(reversed(problem.jobs)),
            problem.capacity,
            ExactLatencySurface(
                {
                    (rows, slots): duration
                    for rows, slots, duration in reversed(problem.latency.entries)
                }
            ),
            wait=problem.wait,
            start_time=problem.start_time,
        )
        reversed_certificate = solve_general_oracle(reversed_problem, limits=_limits())
        if (
            reversed_problem.input_hash != problem.input_hash
            or reversed_certificate.certificate_hash
            != certificate_by_id[scenario_id].certificate_hash
        ):
            raise OracleCampaignError("input-order invariance failed")
        order_invariance += 1
    for scenario_id in sorted(proof_ids)[:12]:
        problem = problem_by_id[scenario_id]
        shifted = _shift_problem(problem, Fraction(7, 3))
        shifted_certificate = solve_general_oracle(shifted, limits=_limits())
        if shifted_certificate.objective != certificate_by_id[scenario_id].objective:
            raise OracleCampaignError("uniform time-shift objective invariance failed")
        original_completions = dict(certificate_by_id[scenario_id].completion_times)
        shifted_completions = dict(shifted_certificate.completion_times)
        if any(
            shifted_completions[job_id] != completion + Fraction(7, 3)
            for job_id, completion in original_completions.items()
        ):
            raise OracleCampaignError("uniform time-shift completion invariance failed")
        shift_invariance += 1
    by_capacity: dict[
        tuple[str, str, Fraction, Fraction, PhysicalMode],
        dict[CapacityMode, GeneralOracleCertificate],
    ] = defaultdict(dict)
    spec_by_id = {spec.scenario_id: spec for spec in specs}
    for scenario_id, certificate in certificate_by_id.items():
        spec = spec_by_id[scenario_id]
        key = (
            spec.arrival_pattern,
            spec.outcome_pattern,
            spec.recovery_eta,
            spec.recovery_jitter,
            spec.physical_mode,
        )
        by_capacity[key][spec.capacity_mode] = certificate
    capacity_pairs = 0
    for capacities in by_capacity.values():
        if set(capacities) != {"tight", "wide"}:
            continue
        if capacities["wide"].objective > capacities["tight"].objective:
            raise OracleCampaignError("capacity monotonicity failed")
        capacity_pairs += 1
    aliases = (
        "work-conserving-edf",
        "saguaro-barrier",
        "spectre-parallel-padded",
        "immediate-fission",
    )
    alias_groups = 0
    row_index = {
        (
            str(row["scenario_id"]),
            str(row["controller_setting"]),
            str(row["policy"]),
        ): row
        for row in rows
    }
    for spec in specs:
        for setting in controller_settings():
            schedules = [
                schedule_by_key[(spec.scenario_id, setting.setting_id, policy)]
                for policy in aliases
            ]
            if any(schedule.events != schedules[0].events for schedule in schedules[1:]):
                raise OracleCampaignError("dispatch-component alias equivalence failed")
            objectives = [
                (
                    row_index[(spec.scenario_id, setting.setting_id, policy)][
                        "candidate_deadline_violations"
                    ],
                    row_index[(spec.scenario_id, setting.setting_id, policy)][
                        "candidate_weighted_flow"
                    ],
                )
                for policy in aliases
            ]
            if any(objective != objectives[0] for objective in objectives[1:]):
                raise OracleCampaignError("dispatch-component alias objective mismatch")
            alias_groups += 1
    return {
        "input_order_and_latency_map_order_invariance_cases": order_invariance,
        "uniform_time_shift_invariance_cases": shift_invariance,
        "capacity_monotonicity_pairs": capacity_pairs,
        "dispatch_component_alias_groups": alias_groups,
        "cancellation_compilations": sum(spec.family == "cancellation" for spec in specs),
        "terminal_failure_compilations": sum(spec.family == "terminal-failure" for spec in specs),
        "retry_compilations": sum(spec.family == "retry" for spec in specs),
    }


@dataclass(frozen=True, slots=True)
class CampaignResult:
    design: JsonObject
    certificates: tuple[JsonObject, ...]
    comparisons: tuple[ComparisonRow, ...]
    summary: JsonObject
    counterexamples: tuple[JsonObject, ...]
    coverage: JsonObject


def run_campaign(config: ModeConfig) -> CampaignResult:
    """Solve every exact problem and evaluate every policy/controller case."""

    specs = campaign_specs(config.mode)
    proof_ids = _proof_ids(specs, min(config.independent_proofs, len(specs)))
    problem_by_id: dict[str, OracleProblem] = {}
    certificate_by_id: dict[str, GeneralOracleCertificate] = {}
    certificate_documents: list[JsonObject] = []
    proof_nodes = 0
    for spec in specs:
        problem = compile_problem(spec)
        if problem.input_hash in {prior.input_hash for prior in problem_by_id.values()}:
            raise OracleCampaignError("scenario matrix compiled duplicate exact inputs")
        certificate = solve_general_oracle(problem, limits=_limits())
        independently_proved = spec.scenario_id in proof_ids
        verification = verify_general_oracle_certificate(
            problem,
            certificate,
            prove_optimality=independently_proved,
            max_verifier_nodes=5_000_000,
        )
        proof_nodes += verification.verifier_nodes
        problem_by_id[spec.scenario_id] = problem
        certificate_by_id[spec.scenario_id] = certificate
        certificate_documents.append(
            {
                "scenario": spec,
                "jobs": problem.jobs,
                "capacity": problem.capacity,
                "latency_surface": problem.latency,
                "wait_config": problem.wait,
                "decision_points": problem.decision_points,
                "problem_hash": problem.input_hash,
                "certificate_hash": certificate.certificate_hash,
                "certificate": certificate,
                "verification": verification,
            }
        )
    rows: list[ComparisonRow] = []
    schedule_by_key: dict[tuple[str, str, str], ScheduleEvaluation] = {}
    for spec in specs:
        problem = problem_by_id[spec.scenario_id]
        certificate = certificate_by_id[spec.scenario_id]
        edf = work_conserving_edf(problem)
        for setting in controller_settings():
            candidates: list[tuple[str, ScheduleEvaluation]] = [("work-conserving-edf", edf)]
            candidates.extend(
                (
                    policy.name,
                    evaluate_dispatch_policy(problem, spec, setting, policy),
                )
                for policy in _policies(setting)
            )
            if tuple(name for name, _ in candidates) != POLICY_NAMES:
                raise OracleCampaignError("policy inventory drifted")
            for policy_name, candidate in candidates:
                schedule_by_key[(spec.scenario_id, setting.setting_id, policy_name)] = candidate
                rows.append(
                    _comparison_row(
                        spec=spec,
                        setting=setting,
                        policy_name=policy_name,
                        candidate=candidate,
                        optimum=certificate,
                    )
                )
    validations = _metamorphic_validations(
        specs=specs,
        problem_by_id=problem_by_id,
        certificate_by_id=certificate_by_id,
        proof_ids=proof_ids,
        rows=rows,
        schedule_by_key=schedule_by_key,
    )
    counterexamples = _counterexample_documents(
        rows,
        schedule_by_key,
        certificate_by_id,
    )
    states = [certificate.states_explored for certificate in certificate_by_id.values()]
    transitions = [certificate.transitions_explored for certificate in certificate_by_id.values()]
    family_counts = Counter(spec.family for spec in specs)
    outcome_counts = Counter(spec.outcome_pattern for spec in specs)
    coverage: JsonObject = {
        "schema_version": SCHEMA_VERSION,
        "mode": config.mode,
        "exact_dynamic_programming_problems": len(specs),
        "unique_problem_hashes": len(
            {certificate.input_hash for certificate in certificate_by_id.values()}
        ),
        "independent_unmemoized_optimality_proofs": len(proof_ids),
        "independent_proof_scenarios": sorted(proof_ids),
        "independent_verifier_nodes": proof_nodes,
        "controller_settings_per_problem": len(controller_settings()),
        "policy_comparisons": len(rows),
        "policies": list(POLICY_NAMES),
        "scenario_families": dict(sorted(family_counts.items())),
        "outcome_patterns": dict(sorted(outcome_counts.items())),
        "arrival_patterns": list(ARRIVAL_PATTERNS),
        "recovery_etas": [_fraction_text(value) for value in RECOVERY_ETAS],
        "recovery_jitters": [_fraction_text(value) for value in RECOVERY_JITTERS],
        "physical_modes": list(PHYSICAL_MODES),
        "capacity_modes": list(CAPACITY_MODES),
        "max_waits": [_fraction_text(value) for value in MAX_WAITS],
        "deadline_guards": [_fraction_text(value) for value in DEADLINE_GUARDS],
        "job_counts": dict(
            sorted(Counter(len(problem.jobs) for problem in problem_by_id.values()).items())
        ),
        "search_accounting": {
            "states_explored_min": min(states),
            "states_explored_max": max(states),
            "states_explored_total": sum(states),
            "transitions_explored_min": min(transitions),
            "transitions_explored_max": max(transitions),
            "transitions_explored_total": sum(transitions),
        },
        "metamorphic_and_adversarial_validations": validations,
        "expansion_over_existing_completion_study": {
            "existing_exact_certificates": 6,
            "campaign_exact_problems": len(specs),
            "factor": len(specs) / 6,
            "nonduplication": (
                "This campaign uses at most three jobs and a full factorial over "
                "pre-realized mechanisms; the prior study used six six-job validation cells."
            ),
        },
    }
    design: JsonObject = {
        "schema_version": SCHEMA_VERSION,
        "study": "Expanded FissionSpec exact scheduler-oracle campaign",
        "mode": config.mode,
        "measurement_warning": WARNING,
        "claim_boundary": CLAIM_BOUNDARY,
        "finite_horizon": {
            "jobs": "two or three active jobs",
            "service": "one non-preemptive target launch per active job",
            "dispatch_actions": (
                "every ordered nonempty released subset satisfying row and slot capacity"
            ),
            "wait_actions": (
                "all releases, exact deadline-safe times, controller guard-safe times, "
                "and release-plus-max-wait grid points"
            ),
            "objective": (
                "lexicographically minimize deadline violations, then exact weighted "
                "flow, then canonical event trace"
            ),
        },
        "pre_realized_mechanisms": {
            "cache_outcomes": [
                "head miss",
                "tail miss",
                "double miss",
            ],
            "recovery": "exact ETA plus deterministic jitter",
            "retry": "one failed remote attempt compiled as 2*ETA+jitter readiness",
            "cancellation": "canceled target row is absent before scheduling starts",
            "terminal_failure": (
                "terminally failed remote row never becomes target-eligible and is absent"
            ),
            "physical_cost": ("packed aggregate slots or next-power-of-two aggregate graph bucket"),
        },
        "controller_factors": {
            "max_wait": [_fraction_text(value) for value in MAX_WAITS],
            "deadline_guard": [_fraction_text(value) for value in DEADLINE_GUARDS],
            "guard_semantics": (
                "controller admission subtracts the guard; the exact objective retains "
                "the original completion deadline"
            ),
        },
        "comparison_scope": {policy: _policy_scope(policy) for policy in POLICY_NAMES},
        "tractability_boundary": {
            "state": "(exact current time, unfinished-job bitmask)",
            "worst_case": (
                "exponential in jobs and optional wait points; permutation actions are "
                "enumerated before safe dominance reduction"
            ),
            "hard_limits": _limits(),
            "excluded": [
                "multi-round token generation",
                "target preemption",
                "live stochastic failures or cancellations after scheduling begins",
                "KV page allocation and cache eviction",
                "remote draft queue scheduling",
                "Saguaro barrier execution cost",
                "SPECTRE padded-recovery execution cost",
                "Rust controller timing or GPU kernels",
            ],
        },
        "selection": "Every predeclared scenario and every policy result is retained.",
        "independent_verification_design": {
            "count": len(proof_ids),
            "selection": (
                "For the full campaign: all 16 outcome-by-physical-by-capacity-by-"
                "two-arrival cells for the two-job cancellation/failure problems, plus "
                "all eight outcome-by-capacity cells for three-job cache/retry problems "
                "under graph-bucket physical widths. This preserves every outcome and "
                "both physical/capacity modes while avoiding the known exponential "
                "packed-width three-job enumeration."
            ),
            "method": (
                "independent recursive enumeration without memoization or permutation-"
                "dominance pruning, capped at 5,000,000 visited verifier nodes per proof"
            ),
        },
    }
    summary: JsonObject = {
        "schema_version": SCHEMA_VERSION,
        "measurement_warning": WARNING,
        "claim_boundary": CLAIM_BOUNDARY,
        "policy_regret_distributions": _policy_summaries(rows),
        "counterexamples_retained": len(counterexamples),
    }
    return CampaignResult(
        design=design,
        certificates=tuple(certificate_documents),
        comparisons=tuple(rows),
        summary=summary,
        counterexamples=tuple(counterexamples),
        coverage=coverage,
    )


COMPARISON_COLUMNS: Final = (
    "schema_version",
    "evidence_class",
    "measurement_warning",
    "scenario_id",
    "family",
    "arrival_pattern",
    "outcome_pattern",
    "recovery_eta",
    "recovery_jitter",
    "physical_mode",
    "capacity_mode",
    "controller_setting",
    "max_wait",
    "deadline_guard",
    "policy",
    "comparison_scope",
    "problem_hash",
    "certificate_hash",
    "oracle_deadline_violations",
    "oracle_weighted_flow",
    "candidate_deadline_violations",
    "candidate_weighted_flow",
    "deadline_violation_gap",
    "weighted_flow_gap",
    "lexicographically_optimal",
    "candidate_trace_sha256",
)
BUNDLE_DATA_FILES: Final = (
    "SUMMARY.md",
    "certificates.jsonl",
    "comparisons.csv",
    "counterexamples.json",
    "coverage.json",
    "design.json",
    "environment.json",
    "summary.json",
)


def _render_json(value: object) -> bytes:
    return (
        json.dumps(
            _jsonable(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        + b"\n"
    )


def _render_jsonl(values: Iterable[object]) -> bytes:
    return b"".join(canonical_json_bytes(value) + b"\n" for value in values)


def _render_comparisons(rows: Sequence[ComparisonRow]) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(
        handle,
        fieldnames=COMPARISON_COLUMNS,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode()


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_render_json(value))


def _write_jsonl(path: Path, values: Iterable[object]) -> None:
    path.write_bytes(_render_jsonl(values))


def _write_comparisons(path: Path, rows: Sequence[ComparisonRow]) -> None:
    path.write_bytes(_render_comparisons(rows))


def _summary_markdown(result: CampaignResult) -> str:
    summaries = cast(
        list[JsonObject],
        result.summary["policy_regret_distributions"],
    )
    lines = [
        "# Expanded exact CPU scheduler-oracle campaign",
        "",
        f"> {WARNING}",
        "",
        CLAIM_BOUNDARY,
        "",
        (
            f"The campaign solved {result.coverage['exact_dynamic_programming_problems']} "
            "distinct exact problems and retained "
            f"{result.coverage['policy_comparisons']} policy comparisons."
        ),
        "",
        "| policy | optimal / cases | deadline-regret cases | max deadline gap | "
        "p95 same-deadline flow regret |",
        "|---|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        flow = cast(JsonObject, summary["same-deadline-flow-regret"])
        lines.append(
            f"| {summary['policy']} | {summary['lexicographically_optimal_cases']} / "
            f"{summary['cases']} | {summary['deadline_regret_cases']} | "
            f"{summary['maximum_deadline_violation_gap']} | {flow['p95']} |"
        )
    lines.extend(
        (
            "",
            "Saguaro and SPECTRE rows compare only their immediate-dispatch timing",
            "component; their defining barrier and padded-recovery execution costs are",
            "outside this oracle. Counterexamples and null cases are both retained.",
            "",
        )
    )
    return "\n".join(lines)


def _source_hashes(repo_root: Path) -> dict[str, str]:
    return {
        relative: hashlib.sha256((repo_root / relative).read_bytes()).hexdigest()
        for relative in IMPLEMENTATION_PATHS
    }


def _environment(mode: StudyMode) -> JsonObject:
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
            "command_template": [
                "python",
                "experiments/run_oracle_campaign.py",
                "--mode",
                mode,
                "--output-dir",
                "<OUTPUT_DIR>",
            ],
        },
    }


def _semantic_payloads(
    result: CampaignResult,
    mode: StudyMode,
) -> dict[str, bytes]:
    payloads = {
        "SUMMARY.md": _summary_markdown(result).encode(),
        "certificates.jsonl": _render_jsonl(result.certificates),
        "comparisons.csv": _render_comparisons(result.comparisons),
        "counterexamples.json": _render_json(result.counterexamples),
        "coverage.json": _render_json(result.coverage),
        "design.json": _render_json(result.design),
        "environment.json": _render_json(_environment(mode)),
        "summary.json": _render_json(result.summary),
    }
    if tuple(sorted(payloads)) != tuple(sorted(BUNDLE_DATA_FILES)):
        raise OracleCampaignError("internal artifact payload inventory mismatch")
    return payloads


def write_bundle(
    output_dir: Path,
    config: ModeConfig,
    *,
    repo_root: Path,
) -> JsonObject:
    """Generate and hash one complete campaign bundle."""

    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_campaign(config)
    payloads = _semantic_payloads(result, config.mode)
    for filename, payload in payloads.items():
        (output_dir / filename).write_bytes(payload)
    source_hashes = _source_hashes(repo_root)
    manifest: JsonObject = {
        "schema_version": SCHEMA_VERSION,
        "study": "Expanded FissionSpec exact scheduler-oracle campaign",
        "mode": config.mode,
        "measurement_warning": WARNING,
        "claim_boundary": CLAIM_BOUNDARY,
        "source_hashes": source_hashes,
        "source_tree_sha256": sha256_document(source_hashes),
        "artifact_files": {
            filename: {
                "bytes": (output_dir / filename).stat().st_size,
                "sha256": hashlib.sha256((output_dir / filename).read_bytes()).hexdigest(),
            }
            for filename in BUNDLE_DATA_FILES
        },
        "exact_problems": result.coverage["exact_dynamic_programming_problems"],
        "policy_comparisons": result.coverage["policy_comparisons"],
    }
    _write_json(output_dir / "manifest.json", manifest)
    verify_bundle(
        output_dir,
        expected_mode=config.mode,
        repo_root=repo_root,
        semantic=False,
    )
    return manifest


def _unique_json_object(pairs: list[tuple[str, object]]) -> JsonObject:
    result: JsonObject = {}
    for key, value in pairs:
        if key in result:
            raise OracleCampaignError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise OracleCampaignError(f"non-finite JSON number is forbidden: {value}")


def _strict_json(payload: bytes, *, label: str) -> object:
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OracleCampaignError(f"invalid strict JSON: {label}") from exc


def _require_object(value: object, *, label: str) -> JsonObject:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise OracleCampaignError(f"{label} must be a JSON object")
    return cast(JsonObject, value)


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_environment_document(document: object, *, mode: StudyMode) -> None:
    environment = _require_object(document, label="environment.json")
    if set(environment) != {
        "schema_version",
        "measurement_warning",
        "observed_runtime",
        "reproduction_contract",
    }:
        raise OracleCampaignError("environment.json keys differ from schema")
    if (
        environment.get("schema_version") != SCHEMA_VERSION
        or environment.get("measurement_warning") != WARNING
    ):
        raise OracleCampaignError("environment.json metadata mismatch")
    observed = _require_object(
        environment.get("observed_runtime"),
        label="environment.json observed_runtime",
    )
    if set(observed) != {
        "python_version",
        "python_implementation",
        "platform",
        "machine",
        "processor",
        "logical_cpu_count",
    }:
        raise OracleCampaignError("observed runtime keys differ from schema")
    for key in (
        "python_version",
        "python_implementation",
        "platform",
        "machine",
        "processor",
    ):
        if not isinstance(observed.get(key), str):
            raise OracleCampaignError(f"observed runtime field must be a string: {key}")
    cpu_count = observed.get("logical_cpu_count")
    if cpu_count is not None and (
        isinstance(cpu_count, bool) or not isinstance(cpu_count, int) or cpu_count <= 0
    ):
        raise OracleCampaignError("observed logical CPU count must be positive or null")
    reproduction = _require_object(
        environment.get("reproduction_contract"),
        label="environment.json reproduction_contract",
    )
    expected_reproduction: JsonObject = {
        "archival_python_version": "3.12.8",
        "dependency_lock": "requirements/repro.lock",
        "command_template": [
            "python",
            "experiments/run_oracle_campaign.py",
            "--mode",
            mode,
            "--output-dir",
            "<OUTPUT_DIR>",
        ],
    }
    if reproduction != expected_reproduction:
        raise OracleCampaignError("environment reproduction contract mismatch")


def verify_bundle(
    output_dir: Path,
    *,
    expected_mode: StudyMode | None = None,
    repo_root: Path | None = None,
    semantic: bool = True,
) -> JsonObject:
    """Verify closed membership, provenance, strict shapes, and campaign semantics."""

    if output_dir.is_symlink() or not output_dir.is_dir():
        raise OracleCampaignError("campaign bundle must be a non-symlink directory")
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise OracleCampaignError("manifest must be a regular non-symlink file")
    try:
        manifest = _require_object(
            _strict_json(manifest_path.read_bytes(), label="manifest.json"),
            label="manifest.json",
        )
    except OSError as exc:
        raise OracleCampaignError("cannot read oracle campaign manifest") from exc
    expected_manifest_keys = {
        "schema_version",
        "study",
        "mode",
        "measurement_warning",
        "claim_boundary",
        "source_hashes",
        "source_tree_sha256",
        "artifact_files",
        "exact_problems",
        "policy_comparisons",
    }
    if set(manifest) != expected_manifest_keys:
        raise OracleCampaignError("oracle campaign manifest keys differ from schema")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise OracleCampaignError("oracle campaign manifest schema mismatch")
    if manifest.get("study") != "Expanded FissionSpec exact scheduler-oracle campaign":
        raise OracleCampaignError("oracle campaign study identifier mismatch")
    if manifest.get("measurement_warning") != WARNING:
        raise OracleCampaignError("oracle campaign measurement warning mismatch")
    if manifest.get("claim_boundary") != CLAIM_BOUNDARY:
        raise OracleCampaignError("oracle campaign claim boundary mismatch")
    mode = manifest.get("mode")
    if mode not in {"ci", "full"}:
        raise OracleCampaignError("oracle campaign manifest mode is invalid")
    if expected_mode is not None and mode != expected_mode:
        raise OracleCampaignError(f"expected mode {expected_mode!r}, found {mode!r}")
    files = _require_object(manifest.get("artifact_files"), label="artifact_files")
    if set(files) != set(BUNDLE_DATA_FILES):
        raise OracleCampaignError("artifact_files keys differ from frozen inventory")
    expected_entries = set(BUNDLE_DATA_FILES) | {"manifest.json"}
    entries = tuple(output_dir.iterdir())
    actual_entries = {path.name for path in entries}
    if actual_entries != expected_entries:
        raise OracleCampaignError(
            "bundle entries differ from the closed manifest: "
            f"unexpected={sorted(actual_entries - expected_entries)}, "
            f"missing={sorted(expected_entries - actual_entries)}"
        )
    for path in entries:
        if path.is_symlink() or not path.is_file():
            raise OracleCampaignError(
                f"bundle member must be a regular non-symlink file: {path.name}"
            )
    for filename, untyped_record in files.items():
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or "/" in filename
            or "\\" in filename
        ):
            raise OracleCampaignError("artifact filename is not a plain local name")
        record = _require_object(
            untyped_record,
            label=f"artifact_files[{filename!r}]",
        )
        if set(record) != {"bytes", "sha256"}:
            raise OracleCampaignError(f"malformed artifact file record: {filename}")
        byte_count = record.get("bytes")
        digest = record.get("sha256")
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
            or not _valid_sha256(digest)
        ):
            raise OracleCampaignError(f"malformed artifact file record: {filename}")
        payload = (output_dir / filename).read_bytes()
        if hashlib.sha256(payload).hexdigest() != digest:
            raise OracleCampaignError(f"artifact hash mismatch: {filename}")
        if len(payload) != byte_count:
            raise OracleCampaignError(f"artifact size mismatch: {filename}")
    source_hashes = _require_object(
        manifest.get("source_hashes"),
        label="source_hashes",
    )
    if set(source_hashes) != set(IMPLEMENTATION_PATHS) or not all(
        _valid_sha256(value) for value in source_hashes.values()
    ):
        raise OracleCampaignError("campaign source hash inventory is malformed")
    if not _valid_sha256(manifest.get("source_tree_sha256")) or sha256_document(
        source_hashes
    ) != manifest.get("source_tree_sha256"):
        raise OracleCampaignError("campaign source tree hash does not match manifest")
    if repo_root is not None:
        observed_source_hashes = _source_hashes(repo_root)
        if observed_source_hashes != source_hashes:
            raise OracleCampaignError("campaign source hashes do not match")
    config = mode_config(cast(StudyMode, mode))
    scenario_count = len(campaign_specs(config.mode))
    expected_comparisons = scenario_count * len(controller_settings()) * len(POLICY_NAMES)
    try:
        comparison_text = (output_dir / "comparisons.csv").read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise OracleCampaignError("comparison CSV is not UTF-8") from exc
    with io.StringIO(comparison_text, newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        if tuple(reader.fieldnames or ()) != COMPARISON_COLUMNS:
            raise OracleCampaignError("comparison CSV schema mismatch")
        comparison_rows = list(reader)
    if any(
        set(row) != set(COMPARISON_COLUMNS)
        or any(not isinstance(value, str) for value in row.values())
        for row in comparison_rows
    ):
        raise OracleCampaignError("comparison CSV row shape mismatch")
    comparison_count = len(comparison_rows)
    if comparison_count != expected_comparisons:
        raise OracleCampaignError("comparison count differs from frozen design")
    certificate_lines = (output_dir / "certificates.jsonl").read_bytes().splitlines()
    if any(not line for line in certificate_lines):
        raise OracleCampaignError("certificate JSONL contains a blank record")
    certificate_documents = [
        _require_object(
            _strict_json(line, label=f"certificates.jsonl:{index}"),
            label=f"certificates.jsonl:{index}",
        )
        for index, line in enumerate(certificate_lines, start=1)
    ]
    certificate_count = len(certificate_documents)
    if certificate_count != scenario_count:
        raise OracleCampaignError("certificate count differs from frozen design")
    structured_documents: dict[str, type[object]] = {
        "counterexamples.json": list,
        "coverage.json": dict,
        "design.json": dict,
        "summary.json": dict,
    }
    for filename, expected_type in structured_documents.items():
        document = _strict_json((output_dir / filename).read_bytes(), label=filename)
        if not isinstance(document, expected_type):
            raise OracleCampaignError(
                f"{filename} top-level value must be {expected_type.__name__}"
            )
    _validate_environment_document(
        _strict_json(
            (output_dir / "environment.json").read_bytes(),
            label="environment.json",
        ),
        mode=config.mode,
    )
    if (
        manifest.get("exact_problems") != scenario_count
        or manifest.get("policy_comparisons") != expected_comparisons
    ):
        raise OracleCampaignError("manifest campaign counts differ from design")
    if semantic:
        expected_result = run_campaign(config)
        expected_payloads = _semantic_payloads(expected_result, config.mode)
        for filename in BUNDLE_DATA_FILES:
            if filename == "environment.json":
                continue
            if (output_dir / filename).read_bytes() != expected_payloads[filename]:
                raise OracleCampaignError(f"semantic artifact mismatch: {filename}")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("ci", "full"), default="ci")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/oracle_campaign"),
    )
    parser.add_argument("--verify", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    try:
        if args.verify:
            manifest = verify_bundle(
                args.output_dir,
                expected_mode=args.mode,
                repo_root=repo_root,
            )
        else:
            manifest = write_bundle(
                args.output_dir,
                mode_config(args.mode),
                repo_root=repo_root,
            )
    except (OSError, OracleCampaignError, ValueError) as exc:
        print(f"oracle campaign failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
