"""Exact and adversarial tests for token-level speculative semantics."""

from __future__ import annotations

import random
import unittest
from collections import Counter
from dataclasses import replace
from fractions import Fraction

from fissionspec.rng import CounterRNG
from fissionspec.semantics import (
    CommittedState,
    ImpossibleProposalError,
    OutcomeKind,
    SemanticError,
    SessionCompleteError,
    TinyAutoregressiveModel,
    advance_session,
    greedy_speculative_decode,
    greedy_target_decode,
    prepare_outcome_continuations,
    proposal_distribution,
    sample_draft_proposal,
    speculative_decode,
    speculative_sequence_distribution,
    speculative_step,
    start_session,
    target_sequence_distribution,
    verification_outcome_distribution,
)


def _models() -> tuple[TinyAutoregressiveModel, TinyAutoregressiveModel]:
    target = TinyAutoregressiveModel.from_weights(
        3,
        {
            (): (5, 3, 2),
            (0,): (1, 7, 2),
            (1,): (4, 1, 5),
            (2,): (3, 6, 1),
            (0, 1): (7, 2, 1),
        },
    )
    draft = TinyAutoregressiveModel.from_weights(
        3,
        {
            (): (2, 6, 2),
            (0,): (6, 2, 2),
            (1,): (2, 5, 3),
            (2,): (5, 1, 4),
            (0, 1): (1, 1, 8),
        },
    )
    return target, draft


def _random_model(
    source: random.Random,
    vocab_size: int,
    *,
    allow_zeros: bool,
) -> TinyAutoregressiveModel:
    rows: dict[tuple[int, ...], tuple[int, ...]] = {}
    for context in [(), *((token,) for token in range(vocab_size))]:
        lower = 0 if allow_zeros else 1
        weights = tuple(source.randint(lower, 7) for _ in range(vocab_size))
        if sum(weights) == 0:
            weights = (1, *weights[1:])
        rows[context] = weights
    return TinyAutoregressiveModel.from_weights(vocab_size, rows)


class ExactDistributionTests(unittest.TestCase):
    def test_hand_built_models_match_target_for_every_tiny_sequence(self) -> None:
        target, draft = _models()
        for prompt in ((), (0,), (2, 0)):
            for horizon in range(5):
                expected = target_sequence_distribution(target, prompt, horizon)
                self.assertEqual(sum(expected.values(), start=Fraction()), 1)
                for width in range(1, 4):
                    with self.subTest(prompt=prompt, horizon=horizon, width=width):
                        actual = speculative_sequence_distribution(
                            target,
                            draft,
                            prompt,
                            horizon,
                            width,
                        )
                        self.assertEqual(actual, expected)

    def test_random_exact_micro_models_include_draft_support_holes(self) -> None:
        source = random.Random(20260723)
        for case in range(8):
            vocab_size = 2 + case % 2
            target = _random_model(source, vocab_size, allow_zeros=False)
            draft = _random_model(source, vocab_size, allow_zeros=True)
            prompt = () if case % 3 == 0 else (case % vocab_size,)
            for horizon in range(4):
                expected = target_sequence_distribution(target, prompt, horizon)
                for width in (1, 2, 3):
                    with self.subTest(case=case, horizon=horizon, width=width):
                        self.assertEqual(
                            speculative_sequence_distribution(
                                target,
                                draft,
                                prompt,
                                horizon,
                                width,
                            ),
                            expected,
                        )

    def test_fixed_proposal_uses_rejection_residual_and_target_bonus(self) -> None:
        target = TinyAutoregressiveModel.from_weights(2, {(): (3, 1)})
        draft = TinyAutoregressiveModel.from_weights(2, {(): (1, 3)})
        outcomes = verification_outcome_distribution(target, draft, (), (1,))
        by_emitted = {
            outcome.emitted_tokens: probability for outcome, probability in outcomes.items()
        }
        self.assertEqual(
            by_emitted,
            {
                (0,): Fraction(2, 3),
                (1, 0): Fraction(1, 4),
                (1, 1): Fraction(1, 12),
            },
        )
        rejected = next(outcome for outcome in outcomes if outcome.kind is OutcomeKind.REJECTION)
        self.assertEqual(rejected.accepted_draft_tokens, 0)
        self.assertEqual(rejected.outcome_token, 0)

    def test_proposal_distribution_is_autoregressive_and_normalized(self) -> None:
        _, draft = _models()
        distribution = proposal_distribution(draft, (0,), 3)
        self.assertEqual(len(distribution), 27)
        self.assertEqual(sum(distribution.values(), start=Fraction()), 1)
        expected = (
            draft.distribution((0,))[1]
            * draft.distribution((0, 1))[2]
            * draft.distribution((0, 1, 2))[0]
        )
        self.assertEqual(distribution[(1, 2, 0)], expected)


class SampledSemanticsTests(unittest.TestCase):
    def test_deterministic_monte_carlo_tracks_the_exact_joint_distribution(self) -> None:
        target, draft = _models()
        expected = target_sequence_distribution(target, (0,), 2)
        samples = 16_000
        counts: Counter[tuple[int, ...]] = Counter()
        rng = CounterRNG("token-semantics-monte-carlo")
        for request_id in range(samples):
            result = speculative_decode(
                target,
                draft,
                (0,),
                2,
                2,
                rng,
                request_id,
            )
            counts[result.generated_tokens] += 1

        total_variation = Fraction()
        for sequence, probability in expected.items():
            observed = Fraction(counts[sequence], samples)
            total_variation += abs(observed - probability)
        total_variation /= 2
        self.assertLess(float(total_variation), 0.025)
        self.assertEqual(sum(counts.values()), samples)

    def test_greedy_speculation_is_token_identical_to_target(self) -> None:
        target, draft = _models()
        for prompt in ((), (0,), (1, 2)):
            for horizon in range(13):
                expected = greedy_target_decode(target, prompt, horizon)
                for width in (1, 2, 3, 5):
                    with self.subTest(prompt=prompt, horizon=horizon, width=width):
                        self.assertEqual(
                            greedy_speculative_decode(
                                target,
                                draft,
                                prompt,
                                horizon,
                                width,
                            ),
                            expected,
                        )
                        target_state = CommittedState.create(target, (*prompt, *expected))
                        speculative_state = CommittedState.create(
                            target,
                            (
                                *prompt,
                                *greedy_speculative_decode(
                                    target,
                                    draft,
                                    prompt,
                                    horizon,
                                    width,
                                ),
                            ),
                        )
                        self.assertEqual(
                            speculative_state.kv_digest,
                            target_state.kv_digest,
                        )

    def test_committed_digest_excludes_rejected_and_unverified_tokens(self) -> None:
        target = TinyAutoregressiveModel.from_weights(2, {(): (1, 0)})
        draft = TinyAutoregressiveModel.from_weights(2, {(): (0, 1)})
        parent = CommittedState.create(target)
        rng = CounterRNG("forced-rejection")

        short = speculative_step(target, draft, parent, (1,), rng, "request", 0)
        long = speculative_step(target, draft, parent, (1, 1, 1), rng, "request", 0)
        canonical = CommittedState.create(target, (0,))

        self.assertEqual(short.trace.outcome_kind, OutcomeKind.REJECTION)
        self.assertEqual(short.trace.accepted_draft_tokens, 0)
        self.assertEqual(short.state, canonical)
        self.assertEqual(long.state, canonical)
        self.assertEqual(short.state.kv_digest, long.state.kv_digest)
        self.assertNotEqual(
            short.trace.continuation_key,
            long.trace.continuation_key,
        )

    def test_every_round_digest_matches_reconstruction_from_commits(self) -> None:
        target, draft = _models()
        result = speculative_decode(
            target,
            draft,
            (2,),
            11,
            3,
            CounterRNG("digest-reconstruction"),
            "request",
        )
        reconstructed = CommittedState.create(target, (2,))
        for trace in result.traces:
            reconstructed = reconstructed.append(trace.emitted_tokens)
            self.assertEqual(trace.kv_digest, reconstructed.kv_digest)
        self.assertEqual(result.state, reconstructed)
        self.assertEqual(result.state.tokens, (2, *result.generated_tokens))


class OutcomeContinuationTests(unittest.TestCase):
    def test_cache_order_cannot_change_keys_or_prepared_tokens(self) -> None:
        target, draft = _models()
        rng = CounterRNG("outcome-cache")
        request_id = "cache-request"
        parent = CommittedState.create(target, (0,))
        proposal = sample_draft_proposal(draft, parent.tokens, 2, rng, request_id, 0)
        outcomes = list(verification_outcome_distribution(target, draft, parent.tokens, proposal))
        generation_limit = len(parent.tokens) + 8

        forward = prepare_outcome_continuations(
            target,
            draft,
            parent,
            proposal,
            rng,
            request_id,
            0,
            generation_token_limit=generation_limit,
            speculation_length=2,
            outcome_order=outcomes,
        )
        reverse = prepare_outcome_continuations(
            target,
            draft,
            parent,
            proposal,
            rng,
            request_id,
            0,
            generation_token_limit=generation_limit,
            speculation_length=2,
            outcome_order=reversed(outcomes),
        )
        self.assertEqual(forward, reverse)
        self.assertEqual(len(forward), len(outcomes))
        self.assertEqual(
            len({entry.key.digest for entry in forward.entries}),
            len(forward),
        )

        realized = speculative_step(
            target,
            draft,
            parent,
            proposal,
            rng,
            request_id,
            0,
        )
        assert realized.trace.continuation_key is not None
        hit = forward.lookup(realized.trace.continuation_key)
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.committed_state, realized.state)
        self.assertEqual(hit.next_round_id, 1)

        remaining = generation_limit - len(realized.state.tokens)
        expected_width = min(2, remaining - 1)
        direct = sample_draft_proposal(
            draft,
            realized.state.tokens,
            expected_width,
            rng,
            request_id,
            1,
        )
        self.assertEqual(hit.proposal_tokens, direct)

    def test_cache_hit_and_uncached_next_round_have_identical_commits(self) -> None:
        target, draft = _models()
        rng = CounterRNG("cache-hit-equivalence")
        request_id = 41
        session = start_session(target, request_id, (1,), 8, 2)
        proposal = sample_draft_proposal(draft, session.state.tokens, 2, rng, request_id, 0)
        cache = prepare_outcome_continuations(
            target,
            draft,
            session.state,
            proposal,
            rng,
            request_id,
            0,
            generation_token_limit=len(session.state.tokens) + session.max_new_tokens,
            speculation_length=session.speculation_length,
        )

        first = advance_session(
            session,
            target,
            draft,
            rng,
            prepared_proposal=proposal,
        )
        key = first.traces[-1].continuation_key
        assert key is not None
        continuation = cache.lookup(key)
        self.assertIsNotNone(continuation)
        assert continuation is not None

        cached = advance_session(
            first,
            target,
            draft,
            rng,
            prepared_proposal=continuation.proposal_tokens,
        )
        uncached = advance_session(first, target, draft, rng)
        self.assertEqual(cached, uncached)
        self.assertEqual(cached.state.kv_digest, uncached.state.kv_digest)


class ScheduleInvarianceTests(unittest.TestCase):
    def test_barrier_immediate_and_rebatched_execution_are_identical(self) -> None:
        target, draft = _models()
        rng = CounterRNG("schedule-invariance")
        requests = {
            "alpha": ((0,), 9, 3),
            "beta": ((1, 2), 8, 2),
            "gamma": ((), 10, 4),
        }
        baseline = {
            request_id: speculative_decode(
                target,
                draft,
                prompt,
                horizon,
                width,
                rng,
                request_id,
            )
            for request_id, (prompt, horizon, width) in requests.items()
        }

        barrier = {
            request_id: start_session(
                target,
                request_id,
                prompt,
                horizon,
                width,
            )
            for request_id, (prompt, horizon, width) in requests.items()
        }
        while not all(session.complete for session in barrier.values()):
            for request_id in barrier:
                if not barrier[request_id].complete:
                    barrier[request_id] = advance_session(
                        barrier[request_id],
                        target,
                        draft,
                        rng,
                    )

        rebatched = {
            request_id: start_session(
                target,
                request_id,
                prompt,
                horizon,
                width,
            )
            for request_id, (prompt, horizon, width) in requests.items()
        }
        order = ("gamma", "alpha", "gamma", "beta", "alpha", "beta")
        cursor = 0
        while not all(session.complete for session in rebatched.values()):
            request_id = order[cursor % len(order)]
            cursor += 1
            if not rebatched[request_id].complete:
                rebatched[request_id] = advance_session(
                    rebatched[request_id],
                    target,
                    draft,
                    rng,
                )

        self.assertEqual(barrier, baseline)
        self.assertEqual(rebatched, baseline)


class ValidationTests(unittest.TestCase):
    def test_invalid_exact_models_fail_loudly(self) -> None:
        invalid_calls = [
            lambda: TinyAutoregressiveModel(2, {(): (Fraction(1, 2), Fraction(1, 3))}),
            lambda: TinyAutoregressiveModel(2, {(0,): (Fraction(1, 2), Fraction(1, 2))}),
            lambda: TinyAutoregressiveModel(2, {(): (0.5, 0.5)}),  # type: ignore[arg-type]
            lambda: TinyAutoregressiveModel.from_weights(2, {(): (0, 0)}),
            lambda: TinyAutoregressiveModel.from_weights(2, {(): (1, -1)}),
            lambda: TinyAutoregressiveModel.from_weights(2, {(): (1, 2, 3)}),
        ]
        for call in invalid_calls:
            with self.subTest(call=call), self.assertRaises(SemanticError):
                call()

    def test_zero_probability_supplied_proposal_is_rejected(self) -> None:
        target = TinyAutoregressiveModel.from_weights(2, {(): (1, 1)})
        draft = TinyAutoregressiveModel.from_weights(2, {(): (1, 0)})
        with self.assertRaises(ImpossibleProposalError):
            speculative_step(
                target,
                draft,
                CommittedState.create(target),
                (1,),
                CounterRNG("invalid"),
                "request",
                0,
            )

    def test_corrupt_state_and_invalid_session_transitions_are_rejected(self) -> None:
        target, draft = _models()
        state = CommittedState.create(target, (0,))
        with self.assertRaises(SemanticError):
            replace(state, kv_digest="0" * 64)

        complete = start_session(target, "done", (), 0, 2)
        with self.assertRaises(SessionCompleteError):
            advance_session(complete, target, draft, CounterRNG("done"))

        active = start_session(target, "active", (), 3, 2)
        with self.assertRaises(SemanticError):
            advance_session(
                active,
                target,
                draft,
                CounterRNG("active"),
                prepared_proposal=(0,),
            )

    def test_outcome_order_must_be_an_exact_permutation(self) -> None:
        target, draft = _models()
        parent = CommittedState.create(target)
        rng = CounterRNG("bad-order")
        proposal = sample_draft_proposal(draft, (), 1, rng, "request", 0)
        with self.assertRaises(SemanticError):
            prepare_outcome_continuations(
                target,
                draft,
                parent,
                proposal,
                rng,
                "request",
                0,
                generation_token_limit=5,
                speculation_length=2,
                outcome_order=(),
            )


if __name__ == "__main__":
    unittest.main()
