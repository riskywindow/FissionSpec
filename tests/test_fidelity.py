"""Executable invariants for the CPU fidelity layer."""

from __future__ import annotations

import random
import unittest
from collections import defaultdict
from itertools import pairwise

from fissionspec.fidelity import (
    ContextCostModel,
    DraftJob,
    DraftJobKind,
    FidelityConfig,
    FidelityRequest,
    OutcomeClass,
    OutcomeTreeCache,
    RemoteDraftConfig,
    RemoteDraftService,
    resolve_request_classes,
    sample_accepted_tokens,
    sample_realized_outcome,
    simulate_fidelity_trace,
)
from fissionspec.profiles import HardwareProfile
from fissionspec.rng import CounterRNG


def _request(
    request_id: str,
    *,
    arrival_ms: float = 0.0,
    prompt_tokens: int = 128,
    output_tokens: int = 8,
    speculation_length: int = 4,
    class_weights: tuple[tuple[str, float], ...] = (("mixed", 1.0),),
    correlation_key: str | None = None,
    priority: int = 0,
) -> FidelityRequest:
    return FidelityRequest(
        request_id=request_id,
        arrival_ms=arrival_ms,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        speculation_length=speculation_length,
        class_weights=class_weights,
        correlation_key=correlation_key,
        priority=priority,
    )


def _job(
    job_id: str,
    *,
    submit_ms: float = 0.0,
    kind: DraftJobKind = DraftJobKind.PRECOMPUTE,
    priority: int = 0,
    context_tokens: int = 128,
    branches: int = 2,
    payload_bytes: int = 64,
) -> DraftJob:
    return DraftJob(
        job_id=job_id,
        request_id=f"request/{job_id}",
        kind=kind,
        submit_ms=submit_ms,
        context_tokens=context_tokens,
        branches=branches,
        payload_bytes=payload_bytes,
        priority=priority,
    )


def _costs() -> ContextCostModel:
    return ContextCostModel(
        prefill_base_ms=0.2,
        prefill_per_token_ms=0.002,
        target_base_ms=0.8,
        target_per_row_ms=0.1,
        target_per_context_token_ms=0.0005,
        target_per_verifier_slot_ms=0.01,
        draft_base_ms=0.25,
        recovery_base_ms=0.8,
        draft_per_row_ms=0.05,
        draft_per_context_token_ms=0.0002,
        draft_per_branch_ms=0.01,
        network_base_ms=0.03,
        network_per_byte_ms=0.0001,
        network_jitter_ms=0.02,
    )


class OutcomeTreeCacheTests(unittest.TestCase):
    def test_page_rounding_lru_and_rejected_replacement_are_exact(self) -> None:
        cache = OutcomeTreeCache(byte_budget=350, page_size_bytes=100)
        first = cache.insert(("a", 0, 0), logical_bytes=1)
        second = cache.insert(("b", 0, 0), logical_bytes=101)
        self.assertEqual((first.allocated_pages, second.allocated_pages), (1, 3))
        self.assertEqual(cache.logical_bytes, 102)
        self.assertEqual(cache.allocated_bytes, 300)

        self.assertTrue(cache.lookup(("a", 0, 0)))
        mutation = cache.insert(("c", 0, 0), logical_bytes=100)
        self.assertEqual(mutation.evicted_keys, (("b", 0, 0),))
        self.assertEqual(cache.keys, (("a", 0, 0), ("c", 0, 0)))
        self.assertEqual(cache.logical_bytes, 101)
        self.assertEqual(cache.allocated_pages, 2)
        self.assertTrue(cache.discard(("c", 0, 0)))
        self.assertFalse(cache.discard(("c", 0, 0)))
        self.assertEqual((cache.logical_bytes, cache.allocated_pages), (1, 1))

        before = (cache.keys, cache.logical_bytes, cache.allocated_pages)
        rejected = cache.insert(("a", 0, 0), logical_bytes=301)
        self.assertFalse(rejected.admitted)
        self.assertEqual(
            (cache.keys, cache.logical_bytes, cache.allocated_pages),
            before,
        )
        cache.audit()

    def test_lru_tie_break_and_replay_are_deterministic(self) -> None:
        def replay() -> tuple[tuple[tuple[str, int, int], ...], int]:
            cache = OutcomeTreeCache(byte_budget=2, page_size_bytes=1)
            cache.insert(("b", 0, 0), logical_bytes=1)
            cache.insert(("a", 0, 0), logical_bytes=1)
            cache.insert(("c", 0, 0), logical_bytes=1)
            cache.audit()
            return cache.keys, cache.peak_allocated_pages

        self.assertEqual(replay(), replay())
        self.assertEqual(replay(), ((("a", 0, 0), ("c", 0, 0)), 2))


class RequestClassTests(unittest.TestCase):
    def setUp(self) -> None:
        self.classes = (
            OutcomeClass("hot", 1.0, (1.0, 0.0)),
            OutcomeClass("cold", 0.0, (0.0, 1.0)),
        )

    def test_correlation_group_shares_latent_class_and_couples_draws(self) -> None:
        requests = (
            _request(
                "a",
                class_weights=(("hot", 0.5), ("cold", 0.5)),
                correlation_key="tenant-7",
            ),
            _request(
                "b",
                class_weights=(("hot", 0.5), ("cold", 0.5)),
                correlation_key="tenant-7",
            ),
        )
        resolved = resolve_request_classes(requests, self.classes, seed="class-seed")
        self.assertEqual(resolved[0].outcome_class, resolved[1].outcome_class)
        for item in resolved:
            if item.outcome_class.class_id == "hot":
                self.assertEqual(sample_accepted_tokens(item, round_id=0, seed=9), 3)
                self.assertEqual(sample_realized_outcome(item, round_id=0, seed=9), 0)
            else:
                self.assertEqual(sample_accepted_tokens(item, round_id=0, seed=9), 0)
                self.assertEqual(sample_realized_outcome(item, round_id=0, seed=9), 1)

    def test_group_and_request_names_are_rng_domain_separated(self) -> None:
        requests = (
            _request(
                "collision",
                class_weights=(("hot", 0.5), ("cold", 0.5)),
            ),
            _request(
                "member",
                class_weights=(("hot", 0.5), ("cold", 0.5)),
                correlation_key="collision",
            ),
        )
        selections = {
            tuple(
                item.outcome_class.class_id
                for item in resolve_request_classes(requests, self.classes, seed=seed)
            )
            for seed in range(64)
        }
        self.assertTrue(any(left != right for left, right in selections))

    def test_incompatible_correlation_groups_are_rejected(self) -> None:
        requests = (
            _request(
                "a",
                class_weights=(("hot", 0.5), ("cold", 0.5)),
                correlation_key="g",
            ),
            _request(
                "b",
                class_weights=(("hot", 0.7), ("cold", 0.3)),
                correlation_key="g",
            ),
        )
        with self.assertRaisesRegex(ValueError, "share class_weights"):
            resolve_request_classes(requests, self.classes, seed=1)


class ContextCostTests(unittest.TestCase):
    def test_every_cost_surface_depends_on_its_physical_inputs(self) -> None:
        costs = _costs()
        self.assertGreater(costs.prefill_ms(1024), costs.prefill_ms(16))
        self.assertGreater(
            costs.target_ms(context_tokens=(1024, 512), verifier_slots=8),
            costs.target_ms(context_tokens=(16, 16), verifier_slots=8),
        )
        self.assertGreater(
            costs.target_ms(context_tokens=(16, 16), verifier_slots=16),
            costs.target_ms(context_tokens=(16, 16), verifier_slots=8),
        )
        self.assertGreater(
            costs.remote_service_ms(
                recovery=True,
                context_tokens=(1024,),
                branches=(4,),
            ),
            costs.remote_service_ms(
                recovery=False,
                context_tokens=(16,),
                branches=(1,),
            ),
        )
        rng = CounterRNG(7)
        small = costs.network_ms(
            payload_bytes=1,
            rng=rng,
            job_id="same",
            attempt=0,
            direction="request",
        )
        large = costs.network_ms(
            payload_bytes=1001,
            rng=rng,
            job_id="same",
            attempt=0,
            direction="request",
        )
        self.assertAlmostEqual(large - small, 0.1)

    def test_reference_costs_reduce_exactly_to_hardware_profile(self) -> None:
        profile = HardwareProfile.linear(
            target_overhead_ms=1.2,
            target_per_row_ms=0.3,
            draft_overhead_ms=0.2,
            draft_per_row_ms=0.07,
            recovery_overhead_ms=0.8,
            recovery_per_row_ms=0.12,
            verifier_slot_ms=0.011,
        )
        costs = ContextCostModel.reference(profile)
        self.assertEqual(costs.prefill_ms(10_000), 0.0)
        self.assertEqual(
            costs.target_ms(context_tokens=(10, 1000), verifier_slots=9),
            profile.target_latency_ms(2, 9),
        )
        self.assertEqual(
            costs.remote_service_ms(
                recovery=False,
                context_tokens=(10, 1000),
                branches=(1, 99),
            ),
            profile.draft_latency_ms(2),
        )
        self.assertEqual(
            costs.remote_service_ms(
                recovery=True,
                context_tokens=(10, 1000),
                branches=(1, 99),
            ),
            profile.draft_latency_ms(2, recovery=True),
        )
        self.assertEqual(
            costs.network_ms(
                payload_bytes=1_000_000,
                rng=CounterRNG(3),
                job_id="j",
                attempt=0,
                direction="request",
            ),
            0.0,
        )


class RemoteDraftServiceTests(unittest.TestCase):
    def assert_service_invariants(
        self,
        jobs: tuple[DraftJob, ...],
        config: RemoteDraftConfig,
        *,
        seed: int = 5,
    ) -> None:
        trace = RemoteDraftService(config, _costs(), seed=seed).run(jobs)
        self.assertLessEqual(trace.queue_peak, config.queue_capacity)
        terminal = set(trace.successful_job_ids) | set(trace.terminal_failed_job_ids)
        self.assertEqual(terminal, {job.job_id for job in jobs})
        self.assertFalse(set(trace.successful_job_ids) & set(trace.terminal_failed_job_ids))
        for batch in trace.batches:
            self.assertLessEqual(len(batch.job_ids), config.max_batch_size)
            self.assertLessEqual(batch.start_ms, batch.end_ms)
        by_worker: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for batch in trace.batches:
            by_worker[batch.worker_id].append((batch.start_ms, batch.end_ms))
        for intervals in by_worker.values():
            intervals.sort()
            for (_, prior_end), (next_start, _) in pairwise(intervals):
                self.assertLessEqual(prior_end, next_start)
        for attempt in trace.attempts:
            self.assertLessEqual(attempt.network_ready_ms, attempt.admitted_ms)
            self.assertLessEqual(attempt.admitted_ms, attempt.service_start_ms)
            self.assertLessEqual(attempt.service_end_ms, attempt.response_ms)

    def test_continuous_batching_multiworker_jitter_and_capacity(self) -> None:
        jobs = tuple(
            _job(
                f"j{index:02d}",
                submit_ms=0.07 * (index % 4),
                kind=(DraftJobKind.RECOVERY if index % 5 == 0 else DraftJobKind.PRECOMPUTE),
                priority=index % 3,
                context_tokens=32 + index * 17,
                branches=1 + index % 4,
                payload_bytes=8 + index * 11,
            )
            for index in range(18)
        )
        config = RemoteDraftConfig(
            workers=3,
            queue_policy="continuous-batching",
            max_batch_size=4,
            batch_window_ms=0.15,
            queue_capacity=5,
            failure_probability=0.0,
        )
        self.assert_service_invariants(jobs, config)
        first = RemoteDraftService(config, _costs(), seed=5).run(jobs)
        second = RemoteDraftService(config, _costs(), seed=5).run(jobs)
        different = RemoteDraftService(config, _costs(), seed=6).run(jobs)
        self.assertEqual(first, second)
        self.assertNotEqual(first, different)
        self.assertGreater(first.backpressured_attempts, 0)

    def test_continuous_policy_batches_only_compatible_job_kinds(self) -> None:
        jobs = tuple(_job(f"j{index}") for index in range(5))
        config = RemoteDraftConfig(
            workers=1,
            queue_policy="continuous-batching",
            max_batch_size=3,
            queue_capacity=5,
        )
        trace = RemoteDraftService(
            config,
            ContextCostModel(draft_base_ms=1.0, draft_per_row_ms=0.1),
            seed=0,
        ).run(jobs)
        self.assertEqual(tuple(len(batch.job_ids) for batch in trace.batches), (3, 2))
        self.assertTrue(
            all(
                all(
                    next(job.kind for job in jobs if job.job_id == job_id) is batch.kind
                    for job_id in batch.job_ids
                )
                for batch in trace.batches
            )
        )

    def test_priority_queue_reorders_waiting_work(self) -> None:
        jobs = (
            _job("blocker", submit_ms=0.0),
            _job("low", submit_ms=0.1, priority=0),
            _job("high", submit_ms=0.1, priority=20),
        )
        config = RemoteDraftConfig(
            workers=1,
            queue_policy="priority",
            max_batch_size=8,
            queue_capacity=2,
        )
        trace = RemoteDraftService(
            config,
            ContextCostModel(draft_base_ms=1.0),
            seed=1,
        ).run(jobs)
        self.assertEqual(
            tuple(batch.job_ids for batch in trace.batches),
            (("blocker",), ("high",), ("low",)),
        )

    def test_backpressure_is_only_finite_queue_admission_delay(self) -> None:
        jobs = tuple(_job(f"j{index}") for index in range(4))
        config = RemoteDraftConfig(
            workers=1,
            queue_policy="priority",
            max_batch_size=1,
            queue_capacity=1,
        )
        trace = RemoteDraftService(
            config,
            ContextCostModel(draft_base_ms=1.0),
            seed=1,
        ).run(jobs, initial_worker_available_ms=(5.0,))
        admitted = {attempt.job_id: attempt.admitted_ms for attempt in trace.attempts}
        self.assertEqual(admitted["j0"], 0.0)
        self.assertGreater(admitted["j1"], 0.0)
        self.assertEqual(trace.queue_peak, 1)
        self.assertEqual(trace.backpressured_attempts, 3)

    def test_failure_retry_and_terminal_accounting(self) -> None:
        config = RemoteDraftConfig(
            workers=1,
            queue_policy="priority",
            queue_capacity=1,
            failure_probability=1.0,
            max_retries=2,
            retry_backoff_ms=0.4,
        )
        trace = RemoteDraftService(config, _costs(), seed=11).run((_job("doomed"),))
        self.assertEqual(tuple(item.attempt for item in trace.attempts), (0, 1, 2))
        self.assertEqual(trace.retries, 2)
        self.assertEqual(trace.successful_job_ids, ())
        self.assertEqual(trace.terminal_failed_job_ids, ("doomed",))
        self.assertTrue(trace.attempts[-1].terminal)
        self.assertTrue(all(not attempt.success for attempt in trace.attempts))
        self.assert_service_invariants((_job("doomed"),), config, seed=11)

    def test_reference_remote_batch_is_exact_existing_draft_curve(self) -> None:
        profile = HardwareProfile.linear()
        jobs = tuple(_job(f"j{index}", payload_bytes=10_000) for index in range(4))
        trace = RemoteDraftService(
            RemoteDraftConfig.reference(max_batch_size=4),
            ContextCostModel.reference(profile),
            seed=99,
        ).run(jobs)
        self.assertEqual(len(trace.batches), 1)
        batch = trace.batches[0]
        self.assertEqual(batch.start_ms, 0.0)
        self.assertEqual(batch.end_ms, profile.draft_latency_ms(4))
        self.assertTrue(all(record.network_ready_ms == 0.0 for record in trace.attempts))
        self.assertTrue(all(record.response_ms == batch.end_ms for record in trace.attempts))


class FidelityTraceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.classes = (
            OutcomeClass("mixed", 0.65, (0.55, 0.3, 0.1, 0.05)),
            OutcomeClass("tail", 0.2, (0.05, 0.1, 0.25, 0.6)),
        )

    def test_heterogeneous_prefill_ttft_output_and_membership_trace(self) -> None:
        requests = (
            _request("a", arrival_ms=0.0, prompt_tokens=16, output_tokens=1),
            _request(
                "b",
                arrival_ms=0.05,
                prompt_tokens=512,
                output_tokens=7,
                speculation_length=3,
            ),
            _request(
                "c",
                arrival_ms=0.1,
                prompt_tokens=64,
                output_tokens=13,
                speculation_length=6,
                class_weights=(("mixed", 0.5), ("tail", 0.5)),
                priority=5,
            ),
        )
        config = FidelityConfig(
            costs=_costs(),
            remote=RemoteDraftConfig(
                workers=2,
                max_batch_size=3,
                batch_window_ms=0.05,
                queue_capacity=3,
            ),
            cache_byte_budget=4096,
            cache_page_size_bytes=64,
            kv_bytes_per_token=8,
            continuation_tokens=4,
            fanout=2,
            target_batch_size=2,
        )
        trace = simulate_fidelity_trace(requests, self.classes, config, seed=71)
        self.assertEqual(trace, simulate_fidelity_trace(requests, self.classes, config, seed=71))
        self.assertEqual(
            [record.prompt_tokens for record in trace.prefills],
            [16, 512, 64],
        )
        for prior, following in pairwise(trace.prefills):
            self.assertLessEqual(prior.end_ms, following.start_ms)
        request_by_id = {request.request_id: request for request in requests}
        result_by_id = {result.request_id: result for result in trace.requests}
        for request_id, result in result_by_id.items():
            request = request_by_id[request_id]
            self.assertAlmostEqual(
                result.ttft_ms,
                result.first_token_ms - request.arrival_ms,
            )
            self.assertEqual(
                result.emitted_tokens + result.output_tokens_remaining,
                request.output_tokens,
            )
            self.assertGreaterEqual(result.emitted_tokens, 1)
            if result.terminal:
                self.assertIsNone(result.cache_hit)
                self.assertIsNone(result.realized_outcome)
                self.assertEqual(result.cached_outcomes, ())
                self.assertIsNone(result.next_ready_ms)
            else:
                self.assertIsNotNone(result.realized_outcome)
                self.assertEqual(
                    result.cache_hit,
                    result.realized_outcome in result.cached_outcomes,
                )
        self.assertEqual(result_by_id["a"].output_tokens_remaining, 0)
        self.assertTrue(result_by_id["a"].terminal)
        self.assertTrue(
            all(
                len(batch.request_ids) <= config.target_batch_size for batch in trace.target_batches
            )
        )
        self.assertLessEqual(trace.cache_allocated_bytes, config.cache_byte_budget)
        self.assertEqual(
            trace.cache_allocated_bytes,
            trace.cache_allocated_pages * config.cache_page_size_bytes,
        )

    def test_late_precompute_is_a_membership_miss_not_a_probability_draw(self) -> None:
        request = _request("late", prompt_tokens=1)
        config = FidelityConfig(
            costs=ContextCostModel(
                target_base_ms=0.1,
                draft_base_ms=10.0,
                recovery_base_ms=0.2,
            ),
            remote=RemoteDraftConfig.reference(max_batch_size=1),
            cache_byte_budget=128,
            cache_page_size_bytes=16,
            kv_bytes_per_token=4,
            continuation_tokens=2,
            fanout=4,
            target_batch_size=1,
        )
        trace = simulate_fidelity_trace((request,), self.classes, config, seed=4)
        result = trace.requests[0]
        self.assertFalse(result.cache_hit)
        self.assertEqual(result.cached_outcomes, ())
        self.assertEqual(trace.stale_precompute_jobs, 1)
        self.assertIsNotNone(trace.recovery_trace)

    def test_small_cache_evictions_change_realized_membership(self) -> None:
        requests = tuple(_request(f"r{index}", prompt_tokens=1) for index in range(4))
        config = FidelityConfig(
            costs=ContextCostModel(
                target_base_ms=2.0,
                draft_base_ms=0.1,
                recovery_base_ms=0.1,
            ),
            remote=RemoteDraftConfig.reference(max_batch_size=4),
            cache_byte_budget=1,
            cache_page_size_bytes=1,
            kv_bytes_per_token=1,
            continuation_tokens=1,
            fanout=1,
            target_batch_size=4,
        )
        deterministic = (OutcomeClass("mixed", 0.65, (1.0,)),)
        trace = simulate_fidelity_trace(requests, deterministic, config, seed=8)
        self.assertGreater(trace.cache_evictions, 0)
        self.assertEqual(sum(result.cache_hit for result in trace.requests), 1)
        for result in trace.requests:
            self.assertEqual(
                result.cache_hit,
                result.realized_outcome in result.cached_outcomes,
            )

    def test_reference_trace_reduces_to_existing_row_slot_abstraction(self) -> None:
        profile = HardwareProfile.linear(
            target_overhead_ms=1.0,
            target_per_row_ms=0.2,
            draft_overhead_ms=0.1,
            draft_per_row_ms=0.02,
            recovery_overhead_ms=0.7,
            recovery_per_row_ms=0.1,
            verifier_slot_ms=0.03,
        )
        requests = (
            _request("a", prompt_tokens=10, speculation_length=3),
            _request("b", prompt_tokens=10_000, speculation_length=5),
        )
        only = (OutcomeClass("mixed", 1.0, (1.0,)),)
        config = FidelityConfig.reference(profile, target_batch_size=2)
        trace = simulate_fidelity_trace(requests, only, config, seed=0)
        self.assertTrue(
            all(prefill.start_ms == prefill.end_ms == 0.0 for prefill in trace.prefills)
        )
        self.assertEqual(len(trace.target_batches), 1)
        target = trace.target_batches[0]
        self.assertEqual(
            target.end_ms - target.start_ms,
            profile.target_latency_ms(2, 8),
        )
        self.assertEqual(len(trace.precompute_trace.batches), 1)
        draft = trace.precompute_trace.batches[0]
        self.assertEqual(
            draft.end_ms - draft.start_ms,
            profile.draft_latency_ms(2),
        )
        self.assertTrue(all(result.cache_hit for result in trace.requests))
        self.assertIsNone(trace.recovery_trace)

    def test_randomized_replays_conserve_capacity_and_tokens(self) -> None:
        generator = random.Random(20260723)
        for seed in range(24):
            requests = tuple(
                _request(
                    f"s{seed}/r{index}",
                    arrival_ms=generator.random() * 2.0,
                    prompt_tokens=generator.randrange(0, 2049),
                    output_tokens=generator.randrange(1, 33),
                    speculation_length=generator.randrange(1, 9),
                    class_weights=(("mixed", 0.7), ("tail", 0.3)),
                    correlation_key=f"group/{index % 3}" if index % 4 == 0 else None,
                    priority=generator.randrange(-2, 4),
                )
                for index in range(generator.randrange(2, 14))
            )
            config = FidelityConfig(
                costs=_costs(),
                remote=RemoteDraftConfig(
                    workers=1 + seed % 3,
                    queue_policy=("continuous-batching" if seed % 2 == 0 else "priority"),
                    max_batch_size=1 + seed % 5,
                    batch_window_ms=0.03 * (seed % 4),
                    queue_capacity=1 + seed % 6,
                    failure_probability=(seed % 4) * 0.1,
                    max_retries=2,
                    retry_backoff_ms=0.1,
                ),
                cache_byte_budget=64 * (1 + seed % 6),
                cache_page_size_bytes=64,
                kv_bytes_per_token=8,
                continuation_tokens=1 + seed % 4,
                fanout=1 + seed % 4,
                target_batch_size=1 + seed % 5,
            )
            with self.subTest(seed=seed):
                first = simulate_fidelity_trace(requests, self.classes, config, seed=seed)
                second = simulate_fidelity_trace(requests, self.classes, config, seed=seed)
                self.assertEqual(first, second)
                self.assertLessEqual(first.cache_allocated_bytes, config.cache_byte_budget)
                self.assertLessEqual(
                    first.precompute_trace.queue_peak,
                    config.remote.queue_capacity,
                )
                for result, request in zip(
                    first.requests,
                    sorted(requests, key=lambda item: item.request_id),
                    strict=True,
                ):
                    self.assertEqual(
                        result.emitted_tokens + result.output_tokens_remaining,
                        request.output_tokens,
                    )
                    if result.terminal:
                        self.assertIsNone(result.cache_hit)
                        self.assertIsNone(result.realized_outcome)
                        self.assertIsNone(result.next_ready_ms)
                    else:
                        self.assertIsNotNone(result.realized_outcome)
                        self.assertEqual(
                            result.cache_hit,
                            result.realized_outcome in result.cached_outcomes,
                        )
                if first.recovery_trace is not None:
                    self.assertLessEqual(
                        first.recovery_trace.queue_peak,
                        config.remote.queue_capacity,
                    )


if __name__ == "__main__":
    unittest.main()
