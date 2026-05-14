"""Workload descriptions for the FissionSpec reference simulator.

The simulator deliberately keeps workload data immutable.  A workload may
therefore be reused across policies with a counter-addressed RNG to obtain a
paired, schedule-independent comparison.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

ProbabilitySchedule = float | tuple[float, ...]


def _finite_non_negative(value: float, field: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class RequestConfig:
    """Immutable input for one generation request.

    ``acceptance_probability`` is the probability that an entire speculative
    block is accepted.  A tuple supplies a per-round trace; rounds beyond the
    tuple use its final value.  This compact model is intentional: it exposes
    the batching externality without conflating it with a particular draft
    model's token-level acceptance implementation.
    """

    request_id: str
    arrival_ms: float = 0.0
    output_tokens: int = 32
    speculation_length: int = 4
    acceptance_probability: ProbabilitySchedule = 0.8
    tbt_slo_ms: float = 50.0
    deadline_ms: float | None = None
    prompt_tokens: int = 0

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        _finite_non_negative(self.arrival_ms, "arrival_ms")
        _finite_non_negative(self.tbt_slo_ms, "tbt_slo_ms")
        if self.deadline_ms is not None:
            _finite_non_negative(self.deadline_ms, "deadline_ms")
            if self.deadline_ms < self.arrival_ms:
                raise ValueError("deadline_ms must not precede arrival_ms")
        if (
            isinstance(self.output_tokens, bool)
            or not isinstance(self.output_tokens, int)
            or self.output_tokens <= 0
        ):
            raise ValueError("output_tokens must be a positive integer")
        if (
            isinstance(self.speculation_length, bool)
            or not isinstance(self.speculation_length, int)
            or self.speculation_length <= 0
        ):
            raise ValueError("speculation_length must be a positive integer")
        if (
            isinstance(self.prompt_tokens, bool)
            or not isinstance(self.prompt_tokens, int)
            or self.prompt_tokens < 0
        ):
            raise ValueError("prompt_tokens must be a non-negative integer")

        probabilities = self.acceptance_probability
        if isinstance(probabilities, tuple):
            if not probabilities:
                raise ValueError("acceptance_probability tuple must not be empty")
            for probability in probabilities:
                self._validate_probability(probability)
        else:
            self._validate_probability(probabilities)

    @staticmethod
    def _validate_probability(probability: float) -> None:
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise TypeError("acceptance probabilities must be real numbers")
        if not math.isfinite(float(probability)) or not 0.0 <= probability <= 1.0:
            raise ValueError("acceptance probabilities must be in [0, 1]")

    def probability_for_round(self, round_id: int) -> float:
        """Return the whole-block hit probability for ``round_id``."""

        if isinstance(round_id, bool) or not isinstance(round_id, int) or round_id < 0:
            raise ValueError("round_id must be a non-negative integer")
        probabilities = self.acceptance_probability
        if isinstance(probabilities, tuple):
            return float(probabilities[min(round_id, len(probabilities) - 1)])
        return float(probabilities)

    @property
    def absolute_deadline_ms(self) -> float:
        """Explicit deadline, or a conservative output-length TBT budget."""

        if self.deadline_ms is not None:
            return self.deadline_ms
        return self.arrival_ms + self.output_tokens * self.tbt_slo_ms


@dataclass(frozen=True, slots=True)
class Workload:
    """A validated collection of uniquely identified requests."""

    requests: tuple[RequestConfig, ...]
    name: str = "workload"

    def __post_init__(self) -> None:
        if not self.requests:
            raise ValueError("a workload must contain at least one request")
        request_ids = [request.request_id for request in self.requests]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("request_id values must be unique within a workload")

    @classmethod
    def from_iterable(
        cls, requests: Iterable[RequestConfig], *, name: str = "workload"
    ) -> Workload:
        """Materialize an iterable as an immutable workload."""

        return cls(tuple(requests), name=name)

    @classmethod
    def homogeneous(
        cls,
        count: int,
        *,
        arrival_interval_ms: float = 0.0,
        output_tokens: int = 32,
        speculation_length: int = 4,
        acceptance_probability: ProbabilitySchedule = 0.8,
        tbt_slo_ms: float = 50.0,
        id_prefix: str = "r",
        name: str = "homogeneous",
    ) -> Workload:
        """Construct a regular synthetic workload useful for experiments."""

        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("count must be a positive integer")
        _finite_non_negative(arrival_interval_ms, "arrival_interval_ms")
        return cls(
            tuple(
                RequestConfig(
                    request_id=f"{id_prefix}{index}",
                    arrival_ms=index * arrival_interval_ms,
                    output_tokens=output_tokens,
                    speculation_length=speculation_length,
                    acceptance_probability=acceptance_probability,
                    tbt_slo_ms=tbt_slo_ms,
                )
                for index in range(count)
            ),
            name=name,
        )

    def __iter__(self) -> Iterator[RequestConfig]:
        return iter(self.requests)

    def __len__(self) -> int:
        return len(self.requests)
