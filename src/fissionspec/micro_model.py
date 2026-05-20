"""No-download neural micro-model for CPU semantic smoke tests.

This module is deliberately small and dependency-free.  It initializes a tiny
recurrent language model from :class:`~fissionspec.rng.CounterRNG`, evaluates
its floating-point logits on CPU, and quantizes every finite-context
distribution to integer mass.  The resulting
:class:`~fissionspec.semantics.TinyAutoregressiveModel` can be consumed by the
exact speculative-sampling oracle.

The quantized model is a semantic integration fixture.  It is not a proxy for
transformer numerical behavior or production-kernel performance.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import TypeAlias

from fissionspec.rng import CounterRNG, Seed
from fissionspec.semantics import (
    TinyAutoregressiveModel,
    greedy_speculative_decode,
    greedy_target_decode,
    speculative_sequence_distribution,
    target_sequence_distribution,
)

Vector: TypeAlias = tuple[float, ...]
Matrix: TypeAlias = tuple[Vector, ...]


def _positive_integer(value: object, *, field: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer at least {minimum}")
    return value


def _positive_finite(value: object, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0.0
    ):
        raise ValueError(f"{field} must be finite and positive")
    return float(value)


@dataclass(frozen=True, slots=True)
class MicroModelConfig:
    """Shape and initialization contract for one recurrent language model."""

    seed: Seed
    vocab_size: int = 3
    hidden_size: int = 5
    context_window: int = 2
    probability_resolution: int = 65_536
    parameter_scale: float = 0.35

    def __post_init__(self) -> None:
        CounterRNG(self.seed)
        vocabulary = _positive_integer(self.vocab_size, field="vocab_size", minimum=2)
        _positive_integer(self.hidden_size, field="hidden_size")
        _positive_integer(self.context_window, field="context_window")
        resolution = _positive_integer(
            self.probability_resolution,
            field="probability_resolution",
        )
        if resolution < vocabulary:
            raise ValueError("probability_resolution must be at least vocab_size")
        _positive_finite(self.parameter_scale, field="parameter_scale")


def _initialized_vector(
    rng: CounterRNG,
    stream: str,
    length: int,
    scale: float,
) -> Vector:
    return tuple(
        (2.0 * rng.uniform("micro-model-parameter", 0, stream, index) - 1.0) * scale
        for index in range(length)
    )


def _initialized_matrix(
    rng: CounterRNG,
    stream: str,
    rows: int,
    columns: int,
    scale: float,
) -> Matrix:
    return tuple(
        tuple(
            (
                2.0
                * rng.uniform(
                    "micro-model-parameter",
                    row,
                    stream,
                    column,
                )
                - 1.0
            )
            * scale
            for column in range(columns)
        )
        for row in range(rows)
    )


@dataclass(frozen=True, init=False, slots=True)
class RandomRecurrentMicroModel:
    """A tiny randomly initialized recurrent LM evaluated with binary64 math."""

    config: MicroModelConfig
    initial_hidden: Vector
    embeddings: Matrix
    recurrent: Matrix
    hidden_bias: Vector
    output: Matrix
    output_bias: Vector

    def __init__(self, config: MicroModelConfig) -> None:
        if not isinstance(config, MicroModelConfig):
            raise TypeError("config must be a MicroModelConfig")
        rng = CounterRNG(config.seed)
        hidden = config.hidden_size
        vocabulary = config.vocab_size
        scale = float(config.parameter_scale)
        object.__setattr__(self, "config", config)
        object.__setattr__(
            self,
            "initial_hidden",
            _initialized_vector(rng, "initial-hidden", hidden, scale),
        )
        object.__setattr__(
            self,
            "embeddings",
            _initialized_matrix(rng, "embedding", vocabulary, hidden, scale),
        )
        object.__setattr__(
            self,
            "recurrent",
            _initialized_matrix(rng, "recurrent", hidden, hidden, scale),
        )
        object.__setattr__(
            self,
            "hidden_bias",
            _initialized_vector(rng, "hidden-bias", hidden, scale),
        )
        object.__setattr__(
            self,
            "output",
            _initialized_matrix(rng, "output", vocabulary, hidden, scale),
        )
        object.__setattr__(
            self,
            "output_bias",
            _initialized_vector(rng, "output-bias", vocabulary, scale),
        )

    def _context(self, prefix: tuple[int, ...]) -> tuple[int, ...]:
        if any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or not 0 <= token < self.config.vocab_size
            for token in prefix
        ):
            raise ValueError("prefix tokens must be integers inside the vocabulary")
        return prefix[-self.config.context_window :]

    def logits(self, prefix: tuple[int, ...]) -> Vector:
        """Evaluate next-token logits after the configured suffix context."""

        context = self._context(prefix)
        state = self.initial_hidden
        for token in context:
            next_state: list[float] = []
            for row in range(self.config.hidden_size):
                activation = self.hidden_bias[row] + self.embeddings[token][row]
                activation += sum(
                    self.recurrent[row][column] * state[column]
                    for column in range(self.config.hidden_size)
                )
                next_state.append(math.tanh(activation))
            state = tuple(next_state)
        return tuple(
            self.output_bias[token]
            + sum(
                self.output[token][column] * state[column]
                for column in range(self.config.hidden_size)
            )
            for token in range(self.config.vocab_size)
        )

    def probabilities(self, prefix: tuple[int, ...]) -> Vector:
        """Return a stable floating-point softmax for diagnostic inspection."""

        logits = self.logits(prefix)
        maximum = max(logits)
        unnormalized = tuple(math.exp(value - maximum) for value in logits)
        total = sum(unnormalized)
        return tuple(value / total for value in unnormalized)

    def exact_weights(self, prefix: tuple[int, ...]) -> tuple[int, ...]:
        """Quantize softmax mass deterministically while keeping full support."""

        probabilities = self.probabilities(prefix)
        vocabulary = self.config.vocab_size
        remaining_mass = self.config.probability_resolution - vocabulary
        scaled = tuple(probability * remaining_mass for probability in probabilities)
        floors = [math.floor(value) + 1 for value in scaled]
        remainder = self.config.probability_resolution - sum(floors)
        order = sorted(
            range(vocabulary),
            key=lambda token: (-(scaled[token] - math.floor(scaled[token])), token),
        )
        for token in order[:remainder]:
            floors[token] += 1
        weights = tuple(floors)
        if sum(weights) != self.config.probability_resolution or min(weights) <= 0:
            raise AssertionError("probability quantization failed")
        return weights

    def to_exact_model(self) -> TinyAutoregressiveModel:
        """Enumerate the finite suffix table consumed by the exact oracle."""

        contexts = itertools.chain.from_iterable(
            itertools.product(range(self.config.vocab_size), repeat=length)
            for length in range(self.config.context_window + 1)
        )
        rows = {tuple(context): self.exact_weights(tuple(context)) for context in contexts}
        return TinyAutoregressiveModel.from_weights(self.config.vocab_size, rows)


@dataclass(frozen=True, slots=True)
class MicroModelSmokeResult:
    """Auditable counts and fingerprints from the no-download smoke program."""

    target_fingerprint: str
    draft_fingerprint: str
    target_context_rows: int
    draft_context_rows: int
    exact_distribution_cases: int
    greedy_cases: int
    evidence_warning: str = (
        "randomly initialized quantized CPU micro-model; not a real-model or GPU result"
    )


def run_micro_model_smoke(seed: Seed = "fissionspec-neural-smoke-v1") -> MicroModelSmokeResult:
    """Build independent neural fixtures and exercise exact/greedy semantics."""

    parent_provenance = CounterRNG(seed).provenance
    target = RandomRecurrentMicroModel(
        MicroModelConfig(seed=f"{parent_provenance}/target")
    ).to_exact_model()
    draft = RandomRecurrentMicroModel(
        MicroModelConfig(seed=f"{parent_provenance}/draft")
    ).to_exact_model()
    exact_cases = 0
    greedy_cases = 0
    prompts = ((), (0,), (1, 2))
    for prompt in prompts:
        for horizon in range(4):
            expected = target_sequence_distribution(target, prompt, horizon)
            for width in (1, 2, 3):
                actual = speculative_sequence_distribution(
                    target,
                    draft,
                    prompt,
                    horizon,
                    width,
                )
                if actual != expected:
                    raise AssertionError("micro-model speculative distribution mismatch")
                exact_cases += 1
                if greedy_speculative_decode(
                    target,
                    draft,
                    prompt,
                    horizon,
                    width,
                ) != greedy_target_decode(target, prompt, horizon):
                    raise AssertionError("micro-model greedy decode mismatch")
                greedy_cases += 1
    return MicroModelSmokeResult(
        target_fingerprint=target.fingerprint,
        draft_fingerprint=draft.fingerprint,
        target_context_rows=len(target.rows),
        draft_context_rows=len(draft.rows),
        exact_distribution_cases=exact_cases,
        greedy_cases=greedy_cases,
    )


__all__ = [
    "MicroModelConfig",
    "MicroModelSmokeResult",
    "RandomRecurrentMicroModel",
    "run_micro_model_smoke",
]
