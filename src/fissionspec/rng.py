"""Schedule-independent random draws for speculative serving experiments.

The usual :class:`random.Random` interface is intentionally stateful: changing
thread interleavings, batching, or an early-exit path changes every subsequent
draw.  That is a particularly unpleasant source of irreproducibility in an
asynchronous serving simulator.  This module instead treats the complete draw
address as the random variable's key::

    (seed, request_id, round_id, stream, draw)

Every result is a pure function of that tuple.  Callers may evaluate keys in
any order, on any thread, and in independently scheduled processes without
changing the answers.  Values are derived with keyed BLAKE2b from Python's
standard library; this is a reproducibility primitive, not a claim that the
API should be used to protect secrets.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Final, TypeAlias

RNGAtom: TypeAlias = int | str | bytes
"""A canonically encodable component of a random draw address."""

Seed: TypeAlias = RNGAtom

_PERSON: Final[bytes] = b"FissionSpecRNGv1"
_SEED_DOMAIN: Final[bytes] = b"fissionspec/rng/seed/v1\x00"
_DRAW_DOMAIN: Final[bytes] = b"fissionspec/rng/draw/v1\x00"
_TWO_POW_53: Final[int] = 1 << 53


class RNGError(ValueError):
    """Base class for invalid counter-RNG inputs."""


class InvalidRNGKey(RNGError):
    """Raised when a draw address cannot be encoded unambiguously."""


def _encode_atom(value: RNGAtom, *, field: str) -> bytes:
    """Return a typed, canonical encoding of one address component.

    Length framing in :func:`_frame` prevents concatenation ambiguities, while
    the leading type byte ensures that ``1``, ``"1"``, and ``b"1"`` name
    different streams.  Booleans are rejected because Python considers them
    integers and silently accepting them tends to hide caller mistakes.
    """

    if isinstance(value, bool):
        raise InvalidRNGKey(f"{field} must not be a bool")
    if isinstance(value, bytes):
        return b"b" + value
    if isinstance(value, str):
        return b"s" + value.encode("utf-8")
    if isinstance(value, int):
        sign = b"-" if value < 0 else b"+"
        magnitude = abs(value)
        width = max(1, (magnitude.bit_length() + 7) // 8)
        return b"i" + sign + magnitude.to_bytes(width, "big")
    raise InvalidRNGKey(f"{field} must be int, str, or bytes; got {type(value).__name__}")


def _frame(value: bytes) -> bytes:
    """Length-prefix a byte string using a fixed-width network-order length."""

    return len(value).to_bytes(8, "big") + value


def _non_negative(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidRNGKey(f"{field} must be an integer")
    if value < 0:
        raise InvalidRNGKey(f"{field} must be non-negative")
    return value


@dataclass(frozen=True, slots=True)
class RNGKey:
    """The complete, immutable address of one deterministic random draw.

    ``stream`` should identify the semantic use of a draw (for example,
    ``"draft-token"`` or ``"network-jitter"``).  ``draw`` is a local counter
    within that stream, not a global mutable counter.
    """

    seed: Seed
    request_id: RNGAtom
    round_id: int
    stream: RNGAtom
    draw: int

    def __post_init__(self) -> None:
        _encode_atom(self.seed, field="seed")
        _encode_atom(self.request_id, field="request_id")
        _encode_atom(self.stream, field="stream")
        _non_negative(self.round_id, field="round_id")
        _non_negative(self.draw, field="draw")

    def uint64(self) -> int:
        """Return this address's unsigned 64-bit value."""

        return CounterRNG(self.seed).uint64(self.request_id, self.round_id, self.stream, self.draw)

    def uniform(self) -> float:
        """Return this address's IEEE-754-friendly uniform value in ``[0, 1)``."""

        return CounterRNG(self.seed).uniform(self.request_id, self.round_id, self.stream, self.draw)

    def bernoulli(self, probability: float) -> bool:
        """Return a deterministic Bernoulli draw with the given probability."""

        return CounterRNG(self.seed).bernoulli(
            probability, self.request_id, self.round_id, self.stream, self.draw
        )


class CounterRNG:
    """A stateless, counter-addressed deterministic random number generator.

    The object stores only a normalized seed key.  Draw methods do not mutate
    it, so sharing one instance between workers is safe without a lock.
    """

    __slots__ = ("_seed", "_seed_key")

    def __init__(self, seed: Seed) -> None:
        encoded_seed = _encode_atom(seed, field="seed")
        self._seed = seed
        # BLAKE2b keys are limited to 64 bytes.  Normalizing through SHA-256
        # also makes arbitrarily long textual seeds cheap to reuse per draw.
        self._seed_key = hashlib.sha256(_SEED_DOMAIN + _frame(encoded_seed)).digest()

    @property
    def seed(self) -> Seed:
        """The seed value supplied at construction time."""

        return self._seed

    @property
    def provenance(self) -> str:
        """Return a stable, non-secret fingerprint for paired-trace checks."""

        return f"fissionspec-counter-rng-v1:{self._seed_key.hex()}"

    def _digest(
        self,
        request_id: RNGAtom,
        round_id: int,
        stream: RNGAtom,
        draw: int,
        *,
        digest_size: int = 8,
    ) -> bytes:
        _non_negative(round_id, field="round_id")
        _non_negative(draw, field="draw")
        request_bytes = _encode_atom(request_id, field="request_id")
        stream_bytes = _encode_atom(stream, field="stream")

        digest = hashlib.blake2b(
            digest_size=digest_size,
            key=self._seed_key,
            person=_PERSON,
        )
        digest.update(_DRAW_DOMAIN)
        digest.update(_frame(request_bytes))
        digest.update(_frame(_encode_atom(round_id, field="round_id")))
        digest.update(_frame(stream_bytes))
        digest.update(_frame(_encode_atom(draw, field="draw")))
        return digest.digest()

    def uint64(
        self,
        request_id: RNGAtom,
        round_id: int,
        stream: RNGAtom,
        draw: int = 0,
    ) -> int:
        """Return an unsigned 64-bit result for a complete draw address."""

        return int.from_bytes(self._digest(request_id, round_id, stream, draw), "big", signed=False)

    def uniform(
        self,
        request_id: RNGAtom,
        round_id: int,
        stream: RNGAtom,
        draw: int = 0,
    ) -> float:
        """Return a deterministic uniform variate in the half-open interval.

        The high 53 bits are used so that every returned value is exactly
        representable by a binary64 float, following the same resolution used
        by high-quality standard PRNG float conversions.
        """

        significand = self.uint64(request_id, round_id, stream, draw) >> 11
        return significand / _TWO_POW_53

    def bernoulli(
        self,
        probability: float,
        request_id: RNGAtom,
        round_id: int,
        stream: RNGAtom,
        draw: int = 0,
    ) -> bool:
        """Return a deterministic Bernoulli variate.

        ``probability`` must be finite and in ``[0, 1]``.  Boundary values are
        handled without consuming a special sentinel draw, while the address
        remains explicit and schedule independent.
        """

        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise RNGError("probability must be a real number")
        p = float(probability)
        if not math.isfinite(p) or not 0.0 <= p <= 1.0:
            raise RNGError("probability must be finite and between 0 and 1")
        # Validate the key even at the boundaries.  This avoids a surprising
        # discrepancy where an invalid address succeeds only for p=0 or p=1.
        value = self.uniform(request_id, round_id, stream, draw)
        if p == 0.0:
            return False
        if p == 1.0:
            return True
        return value < p


def counter_u64(
    seed: Seed,
    request_id: RNGAtom,
    round_id: int,
    stream: RNGAtom,
    draw: int = 0,
) -> int:
    """Functional shorthand for :meth:`CounterRNG.uint64`."""

    return CounterRNG(seed).uint64(request_id, round_id, stream, draw)


def uniform(
    seed: Seed,
    request_id: RNGAtom,
    round_id: int,
    stream: RNGAtom,
    draw: int = 0,
) -> float:
    """Functional shorthand for :meth:`CounterRNG.uniform`."""

    return CounterRNG(seed).uniform(request_id, round_id, stream, draw)


def bernoulli(
    probability: float,
    seed: Seed,
    request_id: RNGAtom,
    round_id: int,
    stream: RNGAtom,
    draw: int = 0,
) -> bool:
    """Functional shorthand for :meth:`CounterRNG.bernoulli`."""

    return CounterRNG(seed).bernoulli(probability, request_id, round_id, stream, draw)


__all__ = [
    "CounterRNG",
    "InvalidRNGKey",
    "RNGAtom",
    "RNGError",
    "RNGKey",
    "Seed",
    "bernoulli",
    "counter_u64",
    "uniform",
]
