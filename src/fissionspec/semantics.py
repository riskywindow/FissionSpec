"""Token-exact CPU semantics for speculative decoding.

The serving simulator intentionally works in counts and costs.  This module is
the small, executable semantic oracle beneath those abstractions: it performs
real autoregressive draft sampling and the rejection/residual correction from
speculative decoding over exact rational distributions.

The implementation is dependency-free and deliberately tiny-model oriented.
It is suitable for exhaustive proofs and deterministic CPU experiments, not
for high-throughput inference.  Every stochastic choice is addressed by
``(seed, request_id, logical_round, semantic_stream)`` through
:class:`~fissionspec.rng.CounterRNG`; evaluating requests, verifier outcomes, or
cache branches in a different order therefore cannot perturb a draw.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from typing import Final, Protocol, TypeAlias

from fissionspec.rng import CounterRNG

Token: TypeAlias = int
TokenSequence: TypeAlias = tuple[Token, ...]
Distribution: TypeAlias = tuple[Fraction, ...]
SequenceDistribution: TypeAlias = dict[TokenSequence, Fraction]
SemanticRequestId: TypeAlias = str | int
ProbabilityInput: TypeAlias = Fraction | int

_MODEL_DOMAIN: Final[bytes] = b"fissionspec/exact-model/v1\x00"
_STATE_DOMAIN: Final[bytes] = b"fissionspec/committed-kv/v1\x00"
_OUTCOME_DOMAIN: Final[bytes] = b"fissionspec/outcome-continuation/v1\x00"
_DRAFT_STREAM: Final[str] = "token-semantics/draft"
_ACCEPT_STREAM: Final[str] = "token-semantics/accept"
_RESIDUAL_STREAM: Final[str] = "token-semantics/residual"
_BONUS_STREAM: Final[str] = "token-semantics/bonus"
_DIRECT_STREAM: Final[str] = "token-semantics/direct-target"


class SemanticError(ValueError):
    """Raised when an exact model or token-semantic operation is invalid."""


class ImpossibleProposalError(SemanticError):
    """Raised when a supplied proposal has zero probability under the draft."""


class SessionCompleteError(SemanticError):
    """Raised when attempting to advance a completed decoding session."""


class _HashSink(Protocol):
    def update(self, value: bytes, /) -> None: ...


def _plain_int(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SemanticError(f"{field} must be an integer")
    if value < minimum:
        raise SemanticError(f"{field} must be at least {minimum}")
    return value


def _request_id(value: object) -> SemanticRequestId:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise SemanticError("request_id must be a str or int (but not bool)")
    return value


def _frame(value: bytes) -> bytes:
    return len(value).to_bytes(8, "big") + value


def _encode_integer(value: int) -> bytes:
    width = max(1, (value.bit_length() + 7) // 8)
    return value.to_bytes(width, "big")


def _encode_request_id(value: SemanticRequestId) -> bytes:
    if isinstance(value, str):
        return b"s" + value.encode("utf-8")
    sign = b"-" if value < 0 else b"+"
    return b"i" + sign + _encode_integer(abs(value))


def _require_hex_digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise SemanticError(f"{field} must be 64 lowercase hexadecimal characters")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as exc:
        raise SemanticError(f"{field} must be hexadecimal") from exc
    if len(decoded) != 32:
        raise SemanticError(f"{field} must encode 256 bits")
    return value


def _tokens(
    values: Iterable[object],
    *,
    vocab_size: int,
    field: str,
) -> TokenSequence:
    result: list[int] = []
    for index, value in enumerate(values):
        token = _plain_int(value, field=f"{field}[{index}]")
        if token >= vocab_size:
            raise SemanticError(f"{field}[{index}]={token} is outside vocabulary [0, {vocab_size})")
        result.append(token)
    return tuple(result)


def _probability(value: object, *, field: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (Fraction, int)):
        raise SemanticError(f"{field} must be an int or Fraction")
    result = Fraction(value)
    if result < 0:
        raise SemanticError(f"{field} must be non-negative")
    return result


def _distribution(values: Sequence[object], *, vocab_size: int, field: str) -> Distribution:
    if len(values) != vocab_size:
        raise SemanticError(f"{field} must contain exactly {vocab_size} probabilities")
    result = tuple(
        _probability(value, field=f"{field}[{index}]") for index, value in enumerate(values)
    )
    if sum(result, start=Fraction()) != 1:
        raise SemanticError(f"{field} must sum exactly to one")
    return result


def _hash_fraction(digest: _HashSink, value: Fraction) -> None:
    digest.update(_frame(_encode_integer(value.numerator)))
    digest.update(_frame(_encode_integer(value.denominator)))


@dataclass(frozen=True, init=False, slots=True)
class TinyAutoregressiveModel:
    """An exact finite-context autoregressive model over integer token IDs.

    ``rows`` maps context suffixes to categorical distributions.  The longest
    suffix matching the current prefix is selected, so the empty context is a
    required fallback.  Full-prefix tables are a special case, while short
    suffix tables make arbitrary-horizon tests compact.

    Probabilities must be :class:`fractions.Fraction` or integers and sum
    exactly to one.  Use :meth:`from_weights` when integer weights are more
    convenient.
    """

    vocab_size: int
    rows: tuple[tuple[TokenSequence, Distribution], ...]
    fingerprint: str

    def __init__(
        self,
        vocab_size: int,
        rows: Mapping[TokenSequence, Sequence[ProbabilityInput]],
    ) -> None:
        size = _plain_int(vocab_size, field="vocab_size", minimum=1)
        if not isinstance(rows, Mapping):
            raise SemanticError("rows must be a mapping from context tuples to distributions")

        canonical: list[tuple[TokenSequence, Distribution]] = []
        for raw_context, raw_row in rows.items():
            if not isinstance(raw_context, tuple):
                raise SemanticError("model contexts must be tuples")
            context = _tokens(raw_context, vocab_size=size, field="context")
            row = _distribution(raw_row, vocab_size=size, field=f"row[{context!r}]")
            canonical.append((context, row))
        canonical.sort(key=lambda item: (len(item[0]), item[0]))
        if not canonical or canonical[0][0] != ():
            raise SemanticError("rows must include the empty fallback context")

        digest = hashlib.sha256()
        digest.update(_MODEL_DOMAIN)
        digest.update(_frame(_encode_integer(size)))
        for context, row in canonical:
            digest.update(_frame(b"".join(_frame(_encode_integer(token)) for token in context)))
            for probability in row:
                _hash_fraction(digest, probability)

        object.__setattr__(self, "vocab_size", size)
        object.__setattr__(self, "rows", tuple(canonical))
        object.__setattr__(self, "fingerprint", digest.hexdigest())

    @classmethod
    def from_weights(
        cls,
        vocab_size: int,
        rows: Mapping[TokenSequence, Sequence[int]],
    ) -> TinyAutoregressiveModel:
        """Construct a model by normalizing non-negative integer row weights."""

        size = _plain_int(vocab_size, field="vocab_size", minimum=1)
        normalized: dict[TokenSequence, Distribution] = {}
        for context, raw_weights in rows.items():
            if len(raw_weights) != size:
                raise SemanticError(f"weights[{context!r}] must contain exactly {size} entries")
            weights = tuple(
                _plain_int(weight, field=f"weights[{context!r}][{index}]")
                for index, weight in enumerate(raw_weights)
            )
            total = sum(weights)
            if total == 0:
                raise SemanticError(f"weights[{context!r}] must have positive total mass")
            normalized[context] = tuple(Fraction(weight, total) for weight in weights)
        return cls(size, normalized)

    def distribution(self, prefix: Sequence[int]) -> Distribution:
        """Return the exact next-token row for ``prefix``."""

        tokens = _tokens(prefix, vocab_size=self.vocab_size, field="prefix")
        best_length = -1
        best_row: Distribution | None = None
        for context, row in self.rows:
            context_length = len(context)
            if (
                context_length <= len(tokens)
                and context_length > best_length
                and (context_length == 0 or tokens[-context_length:] == context)
            ):
                best_length = context_length
                best_row = row
        # Construction requires the empty fallback, so this is unreachable.
        if best_row is None:  # pragma: no cover - defensive invariant
            raise AssertionError("model has no fallback distribution")
        return best_row


def _committed_digest(
    model_fingerprint: str,
    vocab_size: int,
    tokens: TokenSequence,
) -> str:
    digest = hashlib.sha256()
    digest.update(_STATE_DOMAIN)
    digest.update(bytes.fromhex(model_fingerprint))
    digest.update(_frame(_encode_integer(vocab_size)))
    digest.update(_frame(_encode_integer(len(tokens))))
    for token in tokens:
        digest.update(_frame(_encode_integer(token)))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CommittedState:
    """Canonical target-model/token state used as a CPU KV-cache oracle.

    The digest does not pretend to hash real tensor bytes.  It commits to the
    target model identity and the exact ordered token sequence, which is the
    semantic condition under which a deterministic target KV cache must agree.
    Provisional draft and rejected verifier tokens never enter this object.
    """

    model_fingerprint: str
    vocab_size: int
    tokens: TokenSequence
    kv_digest: str

    def __post_init__(self) -> None:
        fingerprint = _require_hex_digest(
            self.model_fingerprint,
            field="model_fingerprint",
        )
        size = _plain_int(self.vocab_size, field="vocab_size", minimum=1)
        tokens = _tokens(self.tokens, vocab_size=size, field="tokens")
        digest = _require_hex_digest(self.kv_digest, field="kv_digest")
        expected = _committed_digest(fingerprint, size, tokens)
        if digest != expected:
            raise SemanticError("kv_digest does not match the committed model/token state")

    @classmethod
    def create(
        cls,
        model: TinyAutoregressiveModel,
        tokens: Sequence[int] = (),
    ) -> CommittedState:
        """Create a validated committed state for a target model and prompt."""

        if not isinstance(model, TinyAutoregressiveModel):
            raise SemanticError("model must be a TinyAutoregressiveModel")
        committed = _tokens(tokens, vocab_size=model.vocab_size, field="tokens")
        return cls(
            model_fingerprint=model.fingerprint,
            vocab_size=model.vocab_size,
            tokens=committed,
            kv_digest=_committed_digest(model.fingerprint, model.vocab_size, committed),
        )

    def append(self, tokens: Sequence[int]) -> CommittedState:
        """Return a new state after atomically committing ``tokens``."""

        suffix = _tokens(tokens, vocab_size=self.vocab_size, field="tokens")
        committed = self.tokens + suffix
        return CommittedState(
            model_fingerprint=self.model_fingerprint,
            vocab_size=self.vocab_size,
            tokens=committed,
            kv_digest=_committed_digest(self.model_fingerprint, self.vocab_size, committed),
        )


class OutcomeKind(StrEnum):
    """How verification produced the terminal token of a speculative step."""

    REJECTION = "rejection"
    ALL_ACCEPTED = "all_accepted"


@dataclass(frozen=True, slots=True)
class VerificationOutcome:
    """One possible result for a fixed draft proposal."""

    kind: OutcomeKind
    accepted_draft_tokens: int
    outcome_token: Token
    emitted_tokens: TokenSequence

    def __post_init__(self) -> None:
        if not isinstance(self.kind, OutcomeKind):
            raise SemanticError("kind must be an OutcomeKind")
        accepted = _plain_int(
            self.accepted_draft_tokens,
            field="accepted_draft_tokens",
        )
        token = _plain_int(self.outcome_token, field="outcome_token")
        emitted = tuple(self.emitted_tokens)
        if not emitted:
            raise SemanticError("a verification outcome must emit at least one token")
        if emitted[-1] != token:
            raise SemanticError("outcome_token must be the final emitted token")
        expected = accepted + 1
        if len(emitted) != expected:
            raise SemanticError(
                "emitted_tokens must contain the accepted draft prefix and one outcome token"
            )


@dataclass(frozen=True, slots=True)
class OutcomeContinuationKey:
    """Collision-resistant identity for a verifier outcome continuation.

    The parent digest, complete proposal, accepted length, correction/bonus
    token, model incarnation, and logical request epoch are all included.
    Consequently two outcomes that happen to emit the same surface token do
    not alias in an eager continuation cache.
    """

    request_id: SemanticRequestId
    round_id: int
    parent_kv_digest: str
    draft_model_fingerprint: str
    proposal_tokens: TokenSequence
    accepted_draft_tokens: int
    outcome_token: Token
    kind: OutcomeKind

    def __post_init__(self) -> None:
        _request_id(self.request_id)
        _plain_int(self.round_id, field="round_id")
        _require_hex_digest(self.parent_kv_digest, field="parent_kv_digest")
        _require_hex_digest(
            self.draft_model_fingerprint,
            field="draft_model_fingerprint",
        )
        proposal = tuple(self.proposal_tokens)
        if not proposal:
            raise SemanticError("proposal_tokens must not be empty")
        for index, token in enumerate(proposal):
            _plain_int(token, field=f"proposal_tokens[{index}]")
        accepted = _plain_int(
            self.accepted_draft_tokens,
            field="accepted_draft_tokens",
        )
        _plain_int(self.outcome_token, field="outcome_token")
        if not isinstance(self.kind, OutcomeKind):
            raise SemanticError("kind must be an OutcomeKind")
        if self.kind is OutcomeKind.REJECTION and accepted >= len(proposal):
            raise SemanticError("a rejection must occur before the proposal is exhausted")
        if self.kind is OutcomeKind.ALL_ACCEPTED and accepted != len(proposal):
            raise SemanticError("all_accepted must accept the complete proposal")

    @property
    def digest(self) -> str:
        """Return the canonical SHA-256 key used by cache implementations."""

        digest = hashlib.sha256()
        digest.update(_OUTCOME_DOMAIN)
        digest.update(_frame(_encode_request_id(self.request_id)))
        digest.update(_frame(_encode_integer(self.round_id)))
        digest.update(bytes.fromhex(self.parent_kv_digest))
        digest.update(bytes.fromhex(self.draft_model_fingerprint))
        digest.update(_frame(self.kind.value.encode("ascii")))
        digest.update(_frame(_encode_integer(self.accepted_draft_tokens)))
        digest.update(_frame(_encode_integer(self.outcome_token)))
        digest.update(_frame(_encode_integer(len(self.proposal_tokens))))
        for token in self.proposal_tokens:
            digest.update(_frame(_encode_integer(token)))
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class DecodeRound:
    """Observable atomic commit from one logical decoding round."""

    round_id: int
    proposal_tokens: TokenSequence
    emitted_tokens: TokenSequence
    accepted_draft_tokens: int
    outcome_kind: OutcomeKind | None
    continuation_key: OutcomeContinuationKey | None
    kv_digest: str

    @property
    def is_direct_target(self) -> bool:
        """Whether this one-token round bypassed speculation at the horizon."""

        return self.outcome_kind is None


@dataclass(frozen=True, slots=True)
class StepResult:
    """Committed state and trace record returned by one decoder step."""

    state: CommittedState
    trace: DecodeRound


def _integer_weights(distribution: Distribution) -> tuple[int, ...]:
    denominator = math.lcm(*(probability.denominator for probability in distribution))
    return tuple(
        probability.numerator * (denominator // probability.denominator)
        for probability in distribution
    )


def _exact_randbelow(
    rng: CounterRNG,
    request_id: SemanticRequestId,
    round_id: int,
    stream: str,
    bound: int,
) -> int:
    """Return an unbiased counter-addressed integer in ``range(bound)``."""

    _plain_int(bound, field="bound", minimum=1)
    if bound == 1:
        return 0
    bits = (bound - 1).bit_length()
    word_count = (bits + 63) // 64
    mask = (1 << bits) - 1
    attempt = 0
    while True:
        candidate = 0
        for word in range(word_count):
            draw = attempt * word_count + word
            candidate = (candidate << 64) | rng.uint64(
                request_id,
                round_id,
                stream,
                draw,
            )
        candidate &= mask
        if candidate < bound:
            return candidate
        attempt += 1


def _sample_distribution(
    distribution: Distribution,
    rng: CounterRNG,
    request_id: SemanticRequestId,
    round_id: int,
    stream: str,
) -> Token:
    weights = _integer_weights(distribution)
    draw = _exact_randbelow(rng, request_id, round_id, stream, sum(weights))
    cumulative = 0
    for token, weight in enumerate(weights):
        cumulative += weight
        if draw < cumulative:
            return token
    raise AssertionError("categorical draw exceeded total mass")  # pragma: no cover


def _sample_bernoulli(
    probability: Fraction,
    rng: CounterRNG,
    request_id: SemanticRequestId,
    round_id: int,
    stream: str,
) -> bool:
    if probability <= 0:
        return False
    if probability >= 1:
        return True
    draw = _exact_randbelow(
        rng,
        request_id,
        round_id,
        stream,
        probability.denominator,
    )
    return draw < probability.numerator


def _residual_distribution(
    target: Distribution,
    draft: Distribution,
) -> Distribution:
    positive = tuple(max(Fraction(), p - q) for p, q in zip(target, draft, strict=True))
    total = sum(positive, start=Fraction())
    if total == 0:
        raise SemanticError("rejection has no positive target-minus-draft residual")
    return tuple(value / total for value in positive)


def sample_draft_proposal(
    draft: TinyAutoregressiveModel,
    prefix: Sequence[int],
    width: int,
    rng: CounterRNG,
    request_id: SemanticRequestId,
    round_id: int,
) -> TokenSequence:
    """Sample a draft block using logical, schedule-independent draw addresses."""

    if not isinstance(draft, TinyAutoregressiveModel):
        raise SemanticError("draft must be a TinyAutoregressiveModel")
    if not isinstance(rng, CounterRNG):
        raise SemanticError("rng must be a CounterRNG")
    request = _request_id(request_id)
    logical_round = _plain_int(round_id, field="round_id")
    proposal_width = _plain_int(width, field="width", minimum=1)
    context = _tokens(prefix, vocab_size=draft.vocab_size, field="prefix")
    proposal: list[int] = []
    for index in range(proposal_width):
        row = draft.distribution(context + tuple(proposal))
        proposal.append(
            _sample_distribution(
                row,
                rng,
                request,
                logical_round,
                f"{_DRAFT_STREAM}/{index}",
            )
        )
    return tuple(proposal)


def _outcome_key(
    *,
    request_id: SemanticRequestId,
    round_id: int,
    parent: CommittedState,
    draft: TinyAutoregressiveModel,
    proposal: TokenSequence,
    outcome: VerificationOutcome,
) -> OutcomeContinuationKey:
    return OutcomeContinuationKey(
        request_id=request_id,
        round_id=round_id,
        parent_kv_digest=parent.kv_digest,
        draft_model_fingerprint=draft.fingerprint,
        proposal_tokens=proposal,
        accepted_draft_tokens=outcome.accepted_draft_tokens,
        outcome_token=outcome.outcome_token,
        kind=outcome.kind,
    )


def speculative_step(
    target: TinyAutoregressiveModel,
    draft: TinyAutoregressiveModel,
    state: CommittedState,
    proposal: Sequence[int],
    rng: CounterRNG,
    request_id: SemanticRequestId,
    round_id: int,
) -> StepResult:
    """Verify one proposal with exact speculative rejection sampling.

    Candidate ``x`` is accepted with ``min(1, p(x) / q(x))``.  On the first
    rejection, one correction token is drawn from normalized ``(p-q)_+``.  If
    all candidates are accepted, one bonus token is drawn from the target.
    Only the accepted prefix plus correction/bonus token is committed.
    """

    _validate_models_and_state(target, draft, state)
    if not isinstance(rng, CounterRNG):
        raise SemanticError("rng must be a CounterRNG")
    request = _request_id(request_id)
    logical_round = _plain_int(round_id, field="round_id")
    proposed = _tokens(proposal, vocab_size=target.vocab_size, field="proposal")
    if not proposed:
        raise SemanticError("proposal must contain at least one token")

    for index, candidate in enumerate(proposed):
        prefix = state.tokens + proposed[:index]
        p = target.distribution(prefix)
        q = draft.distribution(prefix)
        q_candidate = q[candidate]
        if q_candidate == 0:
            raise ImpossibleProposalError(
                f"proposal token {candidate} at index {index} has zero draft probability"
            )
        acceptance = min(Fraction(1), p[candidate] / q_candidate)
        accepted = _sample_bernoulli(
            acceptance,
            rng,
            request,
            logical_round,
            f"{_ACCEPT_STREAM}/{index}",
        )
        if accepted:
            continue

        residual = _residual_distribution(p, q)
        correction = _sample_distribution(
            residual,
            rng,
            request,
            logical_round,
            f"{_RESIDUAL_STREAM}/{index}",
        )
        emitted = (*proposed[:index], correction)
        outcome = VerificationOutcome(
            kind=OutcomeKind.REJECTION,
            accepted_draft_tokens=index,
            outcome_token=correction,
            emitted_tokens=emitted,
        )
        committed = state.append(emitted)
        key = _outcome_key(
            request_id=request,
            round_id=logical_round,
            parent=state,
            draft=draft,
            proposal=proposed,
            outcome=outcome,
        )
        return StepResult(
            state=committed,
            trace=DecodeRound(
                round_id=logical_round,
                proposal_tokens=proposed,
                emitted_tokens=emitted,
                accepted_draft_tokens=index,
                outcome_kind=outcome.kind,
                continuation_key=key,
                kv_digest=committed.kv_digest,
            ),
        )

    bonus_row = target.distribution(state.tokens + proposed)
    bonus = _sample_distribution(
        bonus_row,
        rng,
        request,
        logical_round,
        _BONUS_STREAM,
    )
    emitted = (*proposed, bonus)
    outcome = VerificationOutcome(
        kind=OutcomeKind.ALL_ACCEPTED,
        accepted_draft_tokens=len(proposed),
        outcome_token=bonus,
        emitted_tokens=emitted,
    )
    committed = state.append(emitted)
    key = _outcome_key(
        request_id=request,
        round_id=logical_round,
        parent=state,
        draft=draft,
        proposal=proposed,
        outcome=outcome,
    )
    return StepResult(
        state=committed,
        trace=DecodeRound(
            round_id=logical_round,
            proposal_tokens=proposed,
            emitted_tokens=emitted,
            accepted_draft_tokens=len(proposed),
            outcome_kind=outcome.kind,
            continuation_key=key,
            kv_digest=committed.kv_digest,
        ),
    )


def _validate_models_and_state(
    target: TinyAutoregressiveModel,
    draft: TinyAutoregressiveModel,
    state: CommittedState,
) -> None:
    if not isinstance(target, TinyAutoregressiveModel):
        raise SemanticError("target must be a TinyAutoregressiveModel")
    if not isinstance(draft, TinyAutoregressiveModel):
        raise SemanticError("draft must be a TinyAutoregressiveModel")
    if not isinstance(state, CommittedState):
        raise SemanticError("state must be a CommittedState")
    if target.vocab_size != draft.vocab_size:
        raise SemanticError("target and draft vocabularies must have equal size")
    if state.model_fingerprint != target.fingerprint:
        raise SemanticError("committed state belongs to a different target model")


def proposal_distribution(
    draft: TinyAutoregressiveModel,
    prefix: Sequence[int],
    width: int,
) -> SequenceDistribution:
    """Enumerate the exact distribution of an autoregressive draft block."""

    if not isinstance(draft, TinyAutoregressiveModel):
        raise SemanticError("draft must be a TinyAutoregressiveModel")
    proposal_width = _plain_int(width, field="width", minimum=1)
    context = _tokens(prefix, vocab_size=draft.vocab_size, field="prefix")
    partial: SequenceDistribution = {(): Fraction(1)}
    for _ in range(proposal_width):
        expanded: SequenceDistribution = {}
        for tokens, probability in partial.items():
            row = draft.distribution(context + tokens)
            for token, token_probability in enumerate(row):
                if token_probability == 0:
                    continue
                candidate = (*tokens, token)
                expanded[candidate] = (
                    expanded.get(candidate, Fraction()) + probability * token_probability
                )
        partial = expanded
    return partial


def verification_outcome_distribution(
    target: TinyAutoregressiveModel,
    draft: TinyAutoregressiveModel,
    prefix: Sequence[int],
    proposal: Sequence[int],
) -> dict[VerificationOutcome, Fraction]:
    """Enumerate verifier outcomes conditional on one fixed proposal."""

    if target.vocab_size != draft.vocab_size:
        raise SemanticError("target and draft vocabularies must have equal size")
    context = _tokens(prefix, vocab_size=target.vocab_size, field="prefix")
    proposed = _tokens(proposal, vocab_size=target.vocab_size, field="proposal")
    if not proposed:
        raise SemanticError("proposal must contain at least one token")

    outcomes: dict[VerificationOutcome, Fraction] = {}
    accepted_probability = Fraction(1)
    for index, candidate in enumerate(proposed):
        step_prefix = context + proposed[:index]
        p = target.distribution(step_prefix)
        q = draft.distribution(step_prefix)
        q_candidate = q[candidate]
        if q_candidate == 0:
            raise ImpossibleProposalError(
                f"proposal token {candidate} at index {index} has zero draft probability"
            )
        acceptance = min(Fraction(1), p[candidate] / q_candidate)
        rejection_probability = accepted_probability * (1 - acceptance)
        if rejection_probability:
            residual = _residual_distribution(p, q)
            for correction, correction_probability in enumerate(residual):
                probability = rejection_probability * correction_probability
                if probability == 0:
                    continue
                emitted = (*proposed[:index], correction)
                outcome = VerificationOutcome(
                    kind=OutcomeKind.REJECTION,
                    accepted_draft_tokens=index,
                    outcome_token=correction,
                    emitted_tokens=emitted,
                )
                outcomes[outcome] = outcomes.get(outcome, Fraction()) + probability
        accepted_probability *= acceptance

    if accepted_probability:
        bonus_row = target.distribution(context + proposed)
        for bonus, bonus_probability in enumerate(bonus_row):
            probability = accepted_probability * bonus_probability
            if probability == 0:
                continue
            emitted = (*proposed, bonus)
            outcome = VerificationOutcome(
                kind=OutcomeKind.ALL_ACCEPTED,
                accepted_draft_tokens=len(proposed),
                outcome_token=bonus,
                emitted_tokens=emitted,
            )
            outcomes[outcome] = outcomes.get(outcome, Fraction()) + probability

    if sum(outcomes.values(), start=Fraction()) != 1:
        raise AssertionError("conditional verifier distribution lost probability mass")
    return outcomes


def target_sequence_distribution(
    target: TinyAutoregressiveModel,
    prompt: Sequence[int],
    max_new_tokens: int,
) -> SequenceDistribution:
    """Enumerate target-only output probabilities exactly."""

    if not isinstance(target, TinyAutoregressiveModel):
        raise SemanticError("target must be a TinyAutoregressiveModel")
    context = _tokens(prompt, vocab_size=target.vocab_size, field="prompt")
    horizon = _plain_int(max_new_tokens, field="max_new_tokens")
    frontier: SequenceDistribution = {(): Fraction(1)}
    for _ in range(horizon):
        expanded: SequenceDistribution = {}
        for generated, probability in frontier.items():
            row = target.distribution(context + generated)
            for token, token_probability in enumerate(row):
                if token_probability == 0:
                    continue
                sequence = (*generated, token)
                expanded[sequence] = (
                    expanded.get(sequence, Fraction()) + probability * token_probability
                )
        frontier = expanded
    return frontier


def speculative_sequence_distribution(
    target: TinyAutoregressiveModel,
    draft: TinyAutoregressiveModel,
    prompt: Sequence[int],
    max_new_tokens: int,
    speculation_length: int,
) -> SequenceDistribution:
    """Enumerate complete speculative-decoding outputs exactly."""

    if target.vocab_size != draft.vocab_size:
        raise SemanticError("target and draft vocabularies must have equal size")
    context = _tokens(prompt, vocab_size=target.vocab_size, field="prompt")
    horizon = _plain_int(max_new_tokens, field="max_new_tokens")
    width_limit = _plain_int(speculation_length, field="speculation_length", minimum=1)

    memo: dict[tuple[TokenSequence, int], SequenceDistribution] = {}

    def suffix_distribution(
        committed: TokenSequence,
        remaining: int,
    ) -> SequenceDistribution:
        cache_key = (committed, remaining)
        cached = memo.get(cache_key)
        if cached is not None:
            return cached
        if remaining == 0:
            return {(): Fraction(1)}
        if remaining == 1:
            one_token: SequenceDistribution = {
                (token,): probability
                for token, probability in enumerate(target.distribution(committed))
                if probability
            }
            memo[cache_key] = one_token
            return one_token

        result: SequenceDistribution = {}
        width = min(width_limit, remaining - 1)
        proposals = proposal_distribution(draft, committed, width)
        for proposal, proposal_probability in proposals.items():
            outcomes = verification_outcome_distribution(
                target,
                draft,
                committed,
                proposal,
            )
            for outcome, outcome_probability in outcomes.items():
                emitted = outcome.emitted_tokens
                next_prefix = (*committed, *emitted)
                tails = suffix_distribution(next_prefix, remaining - len(emitted))
                for tail, tail_probability in tails.items():
                    sequence = (*emitted, *tail)
                    result[sequence] = (
                        result.get(sequence, Fraction())
                        + proposal_probability * outcome_probability * tail_probability
                    )
        memo[cache_key] = result
        return result

    return suffix_distribution(context, horizon)


def _argmax(distribution: Distribution) -> int:
    # ``max`` keeps the first item on a tie, giving a stable lowest-token rule.
    return max(range(len(distribution)), key=distribution.__getitem__)


def greedy_target_decode(
    target: TinyAutoregressiveModel,
    prompt: Sequence[int],
    max_new_tokens: int,
) -> TokenSequence:
    """Decode greedily from the target, breaking ties by lowest token ID."""

    context = _tokens(prompt, vocab_size=target.vocab_size, field="prompt")
    horizon = _plain_int(max_new_tokens, field="max_new_tokens")
    generated: list[int] = []
    for _ in range(horizon):
        generated.append(_argmax(target.distribution(context + tuple(generated))))
    return tuple(generated)


def greedy_speculative_decode(
    target: TinyAutoregressiveModel,
    draft: TinyAutoregressiveModel,
    prompt: Sequence[int],
    max_new_tokens: int,
    speculation_length: int,
) -> TokenSequence:
    """Run deterministic temperature-zero speculative verification."""

    if target.vocab_size != draft.vocab_size:
        raise SemanticError("target and draft vocabularies must have equal size")
    context = _tokens(prompt, vocab_size=target.vocab_size, field="prompt")
    horizon = _plain_int(max_new_tokens, field="max_new_tokens")
    width_limit = _plain_int(speculation_length, field="speculation_length", minimum=1)
    generated: list[int] = []
    while len(generated) < horizon:
        remaining = horizon - len(generated)
        if remaining == 1:
            generated.append(_argmax(target.distribution(context + tuple(generated))))
            continue
        width = min(width_limit, remaining - 1)
        base = context + tuple(generated)
        proposal: list[int] = []
        for _ in range(width):
            proposal.append(_argmax(draft.distribution(base + tuple(proposal))))

        for accepted, candidate in enumerate(proposal):
            target_token = _argmax(target.distribution(base + tuple(proposal[:accepted])))
            if candidate != target_token:
                generated.extend(proposal[:accepted])
                generated.append(target_token)
                break
        else:
            generated.extend(proposal)
            generated.append(_argmax(target.distribution(base + tuple(proposal))))
    return tuple(generated)


@dataclass(frozen=True, slots=True)
class DecodeSession:
    """Immutable per-request state that can be advanced in any batch order."""

    request_id: SemanticRequestId
    state: CommittedState
    prompt_length: int
    max_new_tokens: int
    speculation_length: int
    next_round_id: int = 0
    traces: tuple[DecodeRound, ...] = ()

    def __post_init__(self) -> None:
        _request_id(self.request_id)
        prompt_length = _plain_int(self.prompt_length, field="prompt_length")
        _plain_int(self.max_new_tokens, field="max_new_tokens")
        _plain_int(self.speculation_length, field="speculation_length", minimum=1)
        _plain_int(self.next_round_id, field="next_round_id")
        if prompt_length > len(self.state.tokens):
            raise SemanticError("prompt_length cannot exceed committed token count")
        if self.generated_count > self.max_new_tokens:
            raise SemanticError("committed generation exceeds max_new_tokens")

    @property
    def generated_tokens(self) -> TokenSequence:
        return self.state.tokens[self.prompt_length :]

    @property
    def generated_count(self) -> int:
        return len(self.state.tokens) - self.prompt_length

    @property
    def remaining(self) -> int:
        return self.max_new_tokens - self.generated_count

    @property
    def complete(self) -> bool:
        return self.remaining == 0


def start_session(
    target: TinyAutoregressiveModel,
    request_id: SemanticRequestId,
    prompt: Sequence[int],
    max_new_tokens: int,
    speculation_length: int,
) -> DecodeSession:
    """Create an immutable request session."""

    request = _request_id(request_id)
    context = _tokens(prompt, vocab_size=target.vocab_size, field="prompt")
    horizon = _plain_int(max_new_tokens, field="max_new_tokens")
    width = _plain_int(speculation_length, field="speculation_length", minimum=1)
    return DecodeSession(
        request_id=request,
        state=CommittedState.create(target, context),
        prompt_length=len(context),
        max_new_tokens=horizon,
        speculation_length=width,
    )


def advance_session(
    session: DecodeSession,
    target: TinyAutoregressiveModel,
    draft: TinyAutoregressiveModel,
    rng: CounterRNG,
    *,
    prepared_proposal: Sequence[int] | None = None,
) -> DecodeSession:
    """Advance one logical round, optionally consuming a cached proposal."""

    if not isinstance(session, DecodeSession):
        raise SemanticError("session must be a DecodeSession")
    _validate_models_and_state(target, draft, session.state)
    if session.complete:
        raise SessionCompleteError("decoding session is already complete")

    round_id = session.next_round_id
    if session.remaining == 1:
        if prepared_proposal is not None:
            raise SemanticError("the final direct-target round cannot consume a proposal")
        row = target.distribution(session.state.tokens)
        token = _sample_distribution(
            row,
            rng,
            session.request_id,
            round_id,
            _DIRECT_STREAM,
        )
        state = session.state.append((token,))
        trace = DecodeRound(
            round_id=round_id,
            proposal_tokens=(),
            emitted_tokens=(token,),
            accepted_draft_tokens=0,
            outcome_kind=None,
            continuation_key=None,
            kv_digest=state.kv_digest,
        )
    else:
        width = min(session.speculation_length, session.remaining - 1)
        if prepared_proposal is None:
            proposal = sample_draft_proposal(
                draft,
                session.state.tokens,
                width,
                rng,
                session.request_id,
                round_id,
            )
        else:
            proposal = _tokens(
                prepared_proposal,
                vocab_size=draft.vocab_size,
                field="prepared_proposal",
            )
            if len(proposal) != width:
                raise SemanticError(
                    f"prepared_proposal has width {len(proposal)}; expected {width}"
                )
        result = speculative_step(
            target,
            draft,
            session.state,
            proposal,
            rng,
            session.request_id,
            round_id,
        )
        state = result.state
        trace = result.trace

    return DecodeSession(
        request_id=session.request_id,
        state=state,
        prompt_length=session.prompt_length,
        max_new_tokens=session.max_new_tokens,
        speculation_length=session.speculation_length,
        next_round_id=round_id + 1,
        traces=(*session.traces, trace),
    )


def speculative_decode(
    target: TinyAutoregressiveModel,
    draft: TinyAutoregressiveModel,
    prompt: Sequence[int],
    max_new_tokens: int,
    speculation_length: int,
    rng: CounterRNG,
    request_id: SemanticRequestId,
) -> DecodeSession:
    """Run a complete sampled request using the stepwise exact semantics."""

    session = start_session(
        target,
        request_id,
        prompt,
        max_new_tokens,
        speculation_length,
    )
    while not session.complete:
        session = advance_session(session, target, draft, rng)
    return session


@dataclass(frozen=True, slots=True)
class PreparedContinuation:
    """Draft work prepared for one exact verifier outcome."""

    key: OutcomeContinuationKey
    committed_state: CommittedState
    next_round_id: int
    proposal_tokens: TokenSequence
    outcome_probability: Fraction


@dataclass(frozen=True, init=False, slots=True)
class OutcomeContinuationCache:
    """Immutable eager cache indexed by complete verifier outcomes."""

    entries: tuple[PreparedContinuation, ...]

    def __init__(self, entries: Iterable[PreparedContinuation]) -> None:
        canonical = tuple(sorted(entries, key=lambda entry: entry.key.digest))
        digests = [entry.key.digest for entry in canonical]
        if len(digests) != len(set(digests)):
            raise SemanticError("outcome continuation keys must be unique")
        object.__setattr__(self, "entries", canonical)

    def lookup(self, key: OutcomeContinuationKey) -> PreparedContinuation | None:
        """Return a prepared continuation, or ``None`` on a genuine miss."""

        if not isinstance(key, OutcomeContinuationKey):
            raise SemanticError("key must be an OutcomeContinuationKey")
        digest = key.digest
        for entry in self.entries:
            if entry.key.digest == digest and entry.key == key:
                return entry
        return None

    def __len__(self) -> int:
        return len(self.entries)


def prepare_outcome_continuations(
    target: TinyAutoregressiveModel,
    draft: TinyAutoregressiveModel,
    parent: CommittedState,
    proposal: Sequence[int],
    rng: CounterRNG,
    request_id: SemanticRequestId,
    round_id: int,
    *,
    generation_token_limit: int,
    speculation_length: int,
    outcome_order: Iterable[VerificationOutcome] | None = None,
) -> OutcomeContinuationCache:
    """Eagerly draft the next block for every possible verifier outcome.

    ``outcome_order`` exists to emulate arbitrary worker completion order.  It
    must be a permutation of the exact positive-probability outcomes.  Because
    proposal draws use the next logical request/round address, cache contents
    are identical for every such order and match an uncached next-round draft.
    """

    _validate_models_and_state(target, draft, parent)
    request = _request_id(request_id)
    logical_round = _plain_int(round_id, field="round_id")
    token_limit = _plain_int(
        generation_token_limit,
        field="generation_token_limit",
        minimum=len(parent.tokens) + 1,
    )
    width_limit = _plain_int(speculation_length, field="speculation_length", minimum=1)
    proposed = _tokens(proposal, vocab_size=target.vocab_size, field="proposal")
    distribution = verification_outcome_distribution(
        target,
        draft,
        parent.tokens,
        proposed,
    )
    ordered = list(distribution) if outcome_order is None else list(outcome_order)
    if len(ordered) != len(distribution) or set(ordered) != set(distribution):
        raise SemanticError("outcome_order must contain every possible outcome exactly once")

    continuations: list[PreparedContinuation] = []
    for outcome in ordered:
        committed = parent.append(outcome.emitted_tokens)
        if len(committed.tokens) > token_limit:
            raise SemanticError("an outcome exceeds generation_token_limit")
        remaining = token_limit - len(committed.tokens)
        if remaining <= 1:
            next_proposal: TokenSequence = ()
        else:
            next_width = min(width_limit, remaining - 1)
            next_proposal = sample_draft_proposal(
                draft,
                committed.tokens,
                next_width,
                rng,
                request,
                logical_round + 1,
            )
        key = _outcome_key(
            request_id=request,
            round_id=logical_round,
            parent=parent,
            draft=draft,
            proposal=proposed,
            outcome=outcome,
        )
        continuations.append(
            PreparedContinuation(
                key=key,
                committed_state=committed,
                next_round_id=logical_round + 1,
                proposal_tokens=next_proposal,
                outcome_probability=distribution[outcome],
            )
        )
    return OutcomeContinuationCache(continuations)
