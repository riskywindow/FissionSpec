"""Deterministic arrival processes and replayable CPU workload traces."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .rng import CounterRNG
from .workload import ProbabilitySchedule, RequestConfig, Workload

_SCHEMA_VERSION: Final = 1


def _positive_float(value: float, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0.0
    ):
        raise ValueError(f"{field} must be finite and positive")
    return float(value)


def _non_negative_float(value: float, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0.0
    ):
        raise ValueError(f"{field} must be finite and non-negative")
    return float(value)


def _positive_count(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class ArrivalTrace:
    """Immutable arrival times plus sufficient generation provenance."""

    times_ms: tuple[float, ...]
    process: str
    rng_provenance: str
    configuration: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.times_ms:
            raise ValueError("arrival trace must not be empty")
        if not self.process:
            raise ValueError("process must not be empty")
        if not self.rng_provenance:
            raise ValueError("rng_provenance must not be empty")
        if any(not math.isfinite(value) or value < 0.0 for value in self.times_ms):
            raise ValueError("arrival times must be finite and non-negative")
        if any(left > right for left, right in zip(self.times_ms, self.times_ms[1:], strict=False)):
            raise ValueError("arrival times must be nondecreasing")
        if tuple(sorted(self.configuration)) != self.configuration:
            raise ValueError("configuration must be sorted by key")
        if len({key for key, _ in self.configuration}) != len(self.configuration):
            raise ValueError("configuration keys must be unique")

    @property
    def sha256(self) -> str:
        document = {
            "schema_version": _SCHEMA_VERSION,
            "process": self.process,
            "rng_provenance": self.rng_provenance,
            "configuration": dict(self.configuration),
            "times_ms": self.times_ms,
        }
        payload = json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        return hashlib.sha256(payload).hexdigest()


def _configuration(**values: object) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((key, str(value)) for key, value in values.items()))


def _exponential_ms(mean_ms: float, uniform: float) -> float:
    # CounterRNG returns a half-open variate, so log1p never sees -1.
    return -mean_ms * math.log1p(-uniform)


def poisson_arrivals(
    *,
    count: int,
    mean_interarrival_ms: float,
    rng: CounterRNG,
    process_id: str = "poisson",
    start_ms: float = 0.0,
) -> ArrivalTrace:
    """Generate a fixed-count Poisson arrival trace.

    The first request arrives at ``start_ms``. Subsequent gaps are independent
    exponentials addressed by request ordinal, so generation has no mutable RNG
    consumption order.
    """

    count = _positive_count(count, field="count")
    mean = _positive_float(mean_interarrival_ms, field="mean_interarrival_ms")
    start = _non_negative_float(start_ms, field="start_ms")
    if not process_id:
        raise ValueError("process_id must not be empty")
    times = [start]
    for ordinal in range(1, count):
        gap = _exponential_ms(
            mean,
            rng.uniform(process_id, ordinal, "poisson-interarrival", 0),
        )
        times.append(times[-1] + gap)
    return ArrivalTrace(
        tuple(times),
        "poisson",
        rng.provenance,
        _configuration(
            count=count,
            mean_interarrival_ms=mean,
            process_id=process_id,
            start_ms=start,
        ),
    )


def pareto_arrivals(
    *,
    count: int,
    minimum_interarrival_ms: float,
    tail_index: float,
    rng: CounterRNG,
    process_id: str = "pareto",
    start_ms: float = 0.0,
) -> ArrivalTrace:
    """Generate heavy-tailed Pareto inter-arrivals with a finite mean.

    ``tail_index`` must exceed one. The first request arrives at ``start_ms``;
    every subsequent gap is at least ``minimum_interarrival_ms``.
    """

    count = _positive_count(count, field="count")
    minimum = _positive_float(
        minimum_interarrival_ms,
        field="minimum_interarrival_ms",
    )
    alpha = _positive_float(tail_index, field="tail_index")
    if alpha <= 1.0:
        raise ValueError("tail_index must exceed one for finite-mean studies")
    start = _non_negative_float(start_ms, field="start_ms")
    if not process_id:
        raise ValueError("process_id must not be empty")
    times = [start]
    for ordinal in range(1, count):
        uniform = rng.uniform(process_id, ordinal, "pareto-interarrival", 0)
        gap = minimum / (1.0 - uniform) ** (1.0 / alpha)
        times.append(times[-1] + gap)
    return ArrivalTrace(
        tuple(times),
        "pareto",
        rng.provenance,
        _configuration(
            count=count,
            minimum_interarrival_ms=minimum,
            process_id=process_id,
            start_ms=start,
            tail_index=alpha,
        ),
    )


def mmpp_arrivals(
    *,
    count: int,
    arrival_rates_per_ms: tuple[float, float],
    transition_rates_per_ms: tuple[float, float],
    rng: CounterRNG,
    initial_state: int = 0,
    process_id: str = "mmpp",
    start_ms: float = 0.0,
    max_events: int = 1_000_000,
) -> ArrivalTrace:
    """Generate an exact two-state Markov-modulated Poisson process.

    Each state races an arrival clock against its continuous-time Markov-chain
    transition clock. The first request is anchored at ``start_ms`` and the
    race generates the remaining fixed count. This avoids a discretized burst
    approximation while remaining dependency-free.
    """

    count = _positive_count(count, field="count")
    max_events = _positive_count(max_events, field="max_events")
    if initial_state not in (0, 1) or isinstance(initial_state, bool):
        raise ValueError("initial_state must be 0 or 1")
    if len(arrival_rates_per_ms) != 2 or len(transition_rates_per_ms) != 2:
        raise ValueError("MMPP rate tuples must contain exactly two states")
    arrivals = tuple(
        _positive_float(value, field="arrival_rates_per_ms") for value in arrival_rates_per_ms
    )
    transitions = tuple(
        _positive_float(value, field="transition_rates_per_ms") for value in transition_rates_per_ms
    )
    start = _non_negative_float(start_ms, field="start_ms")
    if not process_id:
        raise ValueError("process_id must not be empty")

    times = [start]
    state = initial_state
    now = start
    event_ordinal = 0
    while len(times) < count:
        if event_ordinal >= max_events:
            raise RuntimeError("MMPP generation exceeded max_events")
        arrival_rate = arrivals[state]
        transition_rate = transitions[state]
        total_rate = arrival_rate + transition_rate
        mean_event_gap = 1.0 / total_rate
        now += _exponential_ms(
            mean_event_gap,
            rng.uniform(process_id, event_ordinal, "mmpp-race-time", 0),
        )
        event_choice = rng.uniform(process_id, event_ordinal, "mmpp-race-kind", 0)
        if event_choice < arrival_rate / total_rate:
            times.append(now)
        else:
            state = 1 - state
        event_ordinal += 1

    return ArrivalTrace(
        tuple(times),
        "mmpp-2state",
        rng.provenance,
        _configuration(
            arrival_rates_per_ms=arrivals,
            count=count,
            initial_state=initial_state,
            process_id=process_id,
            start_ms=start,
            transition_rates_per_ms=transitions,
        ),
    )


def workload_from_arrivals(
    arrivals: ArrivalTrace,
    *,
    name: str,
    output_tokens: int = 32,
    prompt_tokens: int = 0,
    speculation_length: int = 4,
    cache_hit_probability: ProbabilitySchedule = 0.8,
    token_acceptance_probability: ProbabilitySchedule = 0.8,
    tbt_slo_ms: float = 50.0,
    id_prefix: str = "r",
) -> Workload:
    """Materialize an arrival trace as a homogeneous simulator workload."""

    if not id_prefix:
        raise ValueError("id_prefix must not be empty")
    return Workload(
        tuple(
            RequestConfig(
                request_id=f"{id_prefix}{ordinal}",
                arrival_ms=arrival_ms,
                output_tokens=output_tokens,
                prompt_tokens=prompt_tokens,
                speculation_length=speculation_length,
                cache_hit_probability=cache_hit_probability,
                token_acceptance_probability=token_acceptance_probability,
                tbt_slo_ms=tbt_slo_ms,
            )
            for ordinal, arrival_ms in enumerate(arrivals.times_ms)
        ),
        name=name,
    )


@dataclass(frozen=True, slots=True)
class LoadedTrace:
    """A replay workload linked to the exact source bytes and split."""

    workload: Workload
    source_sha256: str
    source_rows: int
    selected_rows: int
    split: str | None


def _required(row: dict[str, str], field: str, *, line: int) -> str:
    value = row.get(field)
    if value is None or not value.strip():
        raise ValueError(f"trace line {line}: missing {field}")
    return value.strip()


def _integer(
    row: dict[str, str],
    field: str,
    *,
    line: int,
    default: int | None = None,
) -> int:
    raw = row.get(field, "").strip()
    if not raw:
        if default is None:
            raise ValueError(f"trace line {line}: missing {field}")
        return default
    try:
        return int(raw)
    except ValueError as error:
        raise ValueError(f"trace line {line}: invalid integer {field}") from error


def _real(
    row: dict[str, str],
    field: str,
    *,
    line: int,
    default: float | None = None,
) -> float:
    raw = row.get(field, "").strip()
    if not raw:
        if default is None:
            raise ValueError(f"trace line {line}: missing {field}")
        return default
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"trace line {line}: invalid real {field}") from error
    if not math.isfinite(value):
        raise ValueError(f"trace line {line}: non-finite {field}")
    return value


def _probability_schedule(
    row: dict[str, str],
    field: str,
    *,
    line: int,
    default: float,
) -> ProbabilitySchedule:
    raw = row.get(field, "").strip()
    if not raw:
        return default
    try:
        values = tuple(float(part.strip()) for part in raw.split(";"))
    except ValueError as error:
        raise ValueError(f"trace line {line}: invalid probability schedule {field}") from error
    return values[0] if len(values) == 1 else values


def load_trace_csv(
    path: str | Path,
    *,
    split: str | None = None,
    name: str | None = None,
) -> LoadedTrace:
    """Load the stable, dependency-free replay schema.

    Required columns are ``request_id``, ``arrival_ms``, and ``output_tokens``.
    Optional columns are ``split``, ``prompt_tokens``, ``speculation_length``,
    ``cache_hit_probability``, ``token_acceptance_probability``,
    ``tbt_slo_ms``, and ``deadline_ms``. Probability schedules use semicolons.
    Unknown metadata columns are preserved in the source hash and ignored by
    the simulator.
    """

    source = Path(path)
    payload = source.read_bytes()
    source_hash = hashlib.sha256(payload).hexdigest()
    text = payload.decode("utf-8-sig")
    reader = csv.DictReader(text.splitlines())
    if reader.fieldnames is None:
        raise ValueError("trace CSV must have a header")
    required = {"request_id", "arrival_ms", "output_tokens"}
    missing = required - set(reader.fieldnames)
    if missing:
        raise ValueError(f"trace CSV missing columns: {', '.join(sorted(missing))}")
    if split is not None and (not split or "split" not in reader.fieldnames):
        raise ValueError("selecting a split requires a split column")

    requests: list[RequestConfig] = []
    source_rows = 0
    for line, row in enumerate(reader, start=2):
        source_rows += 1
        row_split = row.get("split", "").strip()
        if split is not None and row_split != split:
            continue
        deadline_raw = row.get("deadline_ms", "").strip()
        deadline = _real(row, "deadline_ms", line=line) if deadline_raw else None
        requests.append(
            RequestConfig(
                request_id=_required(row, "request_id", line=line),
                arrival_ms=_real(row, "arrival_ms", line=line),
                output_tokens=_integer(row, "output_tokens", line=line),
                prompt_tokens=_integer(row, "prompt_tokens", line=line, default=0),
                speculation_length=_integer(
                    row,
                    "speculation_length",
                    line=line,
                    default=4,
                ),
                cache_hit_probability=_probability_schedule(
                    row,
                    "cache_hit_probability",
                    line=line,
                    default=0.8,
                ),
                token_acceptance_probability=_probability_schedule(
                    row,
                    "token_acceptance_probability",
                    line=line,
                    default=0.8,
                ),
                tbt_slo_ms=_real(row, "tbt_slo_ms", line=line, default=50.0),
                deadline_ms=deadline,
            )
        )
    if source_rows == 0:
        raise ValueError("trace CSV contains no data rows")
    if not requests:
        raise ValueError(f"trace split {split!r} contains no rows")
    workload_name = name or f"trace-{source.stem}" + (f"-{split}" if split else "")
    return LoadedTrace(
        Workload(tuple(requests), name=workload_name),
        source_sha256=source_hash,
        source_rows=source_rows,
        selected_rows=len(requests),
        split=split,
    )


__all__ = [
    "ArrivalTrace",
    "LoadedTrace",
    "load_trace_csv",
    "mmpp_arrivals",
    "pareto_arrivals",
    "poisson_arrivals",
    "workload_from_arrivals",
]
