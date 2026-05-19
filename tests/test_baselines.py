"""Executable scheduler baselines and deterministic adversarial counterexamples."""

from __future__ import annotations

import unittest

from fissionspec.baselines import (
    BackgroundDraftJob,
    BaselineCostModel,
    BaselineScheduler,
    DeterministicBaselineSimulator,
    DraftJobKind,
    ExecutionMode,
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
from fissionspec.policies import ImmediateFissionPolicy
from fissionspec.profiles import HardwareProfile
from fissionspec.rng import CounterRNG
from fissionspec.simulator import simulate
from fissionspec.workload import RequestConfig, Workload


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


def _linear_costs(
    *,
    target_slot_ms: float = 0.0,
    starvation_ms: float = 50.0,
    draft_context_ms: float = 0.001,
    draft_token_ms: float = 0.04,
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
            name="baseline-test",
        ),
        draft_context_token_ms=draft_context_ms,
        draft_token_ms=draft_token_ms,
        realignment_base_ms=0.2,
        realignment_per_length_ms=0.01,
        starvation_threshold_ms=starvation_ms,
    )


def _run(
    trace: PreRealizedTrace,
    scheduler: BaselineScheduler,
    *,
    costs: BaselineCostModel | None = None,
    batch: int = 2,
):
    return DeterministicBaselineSimulator(
        trace=trace,
        scheduler=scheduler,
        costs=costs,
        max_batch_size=batch,
    ).run()


class BaselineSemanticTests(unittest.TestCase):
    def test_all_baselines_replay_one_matched_semantic_trace(self) -> None:
        trace = PreRealizedTrace(
            (
                RealizedRequest(
                    "a",
                    arrival_ms=0.0,
                    deadline_ms=100.0,
                    prompt_tokens=8,
                    steps=(
                        _step(3, 1, (10, 11), remote=True, rollback=True),
                        _step(3, 2, (12, 13, 14)),
                    ),
                ),
                RealizedRequest(
                    "b",
                    arrival_ms=0.0,
                    deadline_ms=100.0,
                    prompt_tokens=8,
                    steps=(
                        _step(3, 2, (20, 21, 22)),
                        _step(2, 0, (23,)),
                    ),
                ),
                RealizedRequest(
                    "c",
                    arrival_ms=0.2,
                    deadline_ms=100.0,
                    prompt_tokens=3,
                    steps=(_step(2, 1, (30, 31)),),
                ),
            ),
            name="matched",
        )
        policies = (
            SPECTREHybridScheduler(
                SpectreCalibration(
                    ordinary_round_ms=2.0, parallel_round_ms=1.0, rollback_penalty_ms=2.0
                ),
                priority_burst=2,
                context_compression_factor=0.5,
            ),
            EXSpecSlidingPoolScheduler(window_size=8),
            MyopicSlackScheduler(aging_rate=2.0, starvation_bound_ms=5.0),
        )
        results = tuple(_run(trace, policy) for policy in policies)
        assert_semantic_equivalence(*results)
        for result in results:
            with self.subTest(policy=result.policy_name):
                self.assertEqual(result.semantic_signature, trace.semantic_signature)
                self.assertEqual(
                    {request.request_id for request in result.requests},
                    {"a", "b", "c"},
                )
                self.assertGreater(result.metrics.target_launches, 0)
                self.assertGreater(result.metrics.verifier_slots, 0)
                self.assertGreaterEqual(result.metrics.total_ready_wait_ms, 0.0)

    def test_existing_simulator_result_bridges_without_resampling(self) -> None:
        workload = Workload(
            (
                RequestConfig(
                    "a",
                    output_tokens=7,
                    speculation_length=3,
                    cache_hit_probability=(0.0, 1.0),
                    token_acceptance_probability=0.6,
                    prompt_tokens=11,
                ),
                RequestConfig(
                    "b",
                    output_tokens=6,
                    speculation_length=2,
                    cache_hit_probability=1.0,
                    token_acceptance_probability=0.4,
                    prompt_tokens=5,
                ),
            ),
            name="bridge-source",
        )
        source = simulate(
            workload,
            HardwareProfile.linear(name="bridge"),
            ImmediateFissionPolicy(),
            CounterRNG(9),
            max_batch_size=2,
        )
        trace = PreRealizedTrace.from_simulation(source)
        results = (
            _run(trace, FIFOScheduler()),
            _run(trace, EXSpecSlidingPoolScheduler(window_size=4)),
            _run(
                trace,
                SPECTREHybridScheduler(
                    SpectreCalibration(2.0, 1.0, 2.0),
                    context_compression_factor=0.5,
                ),
            ),
        )
        assert_semantic_equivalence(*results)
        expected_tokens = sum(request.output_tokens for request in workload)
        for result in results:
            self.assertEqual(
                sum(len(request.emitted_tokens) for request in result.requests),
                expected_tokens,
            )


class SpectreAbstractionTests(unittest.TestCase):
    def test_calibrated_threshold_is_an_explicit_cost_comparison(self) -> None:
        calibration = SpectreCalibration(
            ordinary_round_ms=10.0,
            parallel_round_ms=6.0,
            rollback_penalty_ms=8.0,
        )
        self.assertEqual(calibration.critical_rollback_ratio, 0.5)
        at_threshold = calibration.decide(0.5)
        above_threshold = calibration.decide(0.5001)
        self.assertEqual(at_threshold.mode, ExecutionMode.PARALLEL)
        self.assertEqual(above_threshold.mode, ExecutionMode.ORDINARY)
        self.assertEqual(at_threshold.parallel_cost_ms, at_threshold.ordinary_cost_ms)

    def test_speculative_priority_is_nonpreemptive_and_has_fairness_escape(self) -> None:
        common_steps = (
            _step(1, 0, (1,), remote=True, rollback=True),
            _step(1, 0, (2,)),
        )
        trace = PreRealizedTrace(
            (
                RealizedRequest("a", 0.0, 100.0, 0, common_steps),
                RealizedRequest("b", 0.0, 100.0, 0, common_steps),
            ),
            (
                BackgroundDraftJob("native-head", 0.0, 3.0),
                BackgroundDraftJob("native-waiting", 0.05, 0.2),
            ),
        )
        scheduler = SPECTREHybridScheduler(
            SpectreCalibration(
                ordinary_round_ms=1.0, parallel_round_ms=2.0, rollback_penalty_ms=1.0
            ),
            priority_burst=1,
        )
        result = _run(trace, scheduler, costs=_linear_costs(), batch=2)
        jobs = [launch.job_id for launch in result.draft_launches]
        self.assertEqual(jobs[0], "native-head")
        self.assertTrue(jobs[1].startswith("spec:"))
        self.assertEqual(jobs[2], "native-waiting")
        self.assertTrue(jobs[3].startswith("spec:"))
        self.assertEqual(
            [launch.kind for launch in result.draft_launches].count(DraftJobKind.BACKGROUND),
            2,
        )
        # The head job is not preempted even when speculative work arrives.
        self.assertEqual(result.draft_launches[0].end_ms, 3.0)

    def test_prompt_compression_reduces_only_recovery_timing(self) -> None:
        trace = PreRealizedTrace(
            (
                RealizedRequest(
                    "long",
                    0.0,
                    100.0,
                    10_000,
                    (
                        _step(2, 0, (1,), remote=True, rollback=True),
                        _step(2, 1, (2, 3)),
                    ),
                ),
            )
        )
        calibration = SpectreCalibration(1.0, 2.0, 1.0)
        full = _run(
            trace,
            SPECTREHybridScheduler(calibration, context_compression_factor=1.0),
            costs=_linear_costs(draft_context_ms=0.001),
            batch=1,
        )
        compressed = _run(
            trace,
            SPECTREHybridScheduler(calibration, context_compression_factor=0.1),
            costs=_linear_costs(draft_context_ms=0.001),
            batch=1,
        )
        assert_semantic_equivalence(full, compressed)
        full_recovery = next(
            launch for launch in full.draft_launches if launch.kind is DraftJobKind.SPECULATIVE
        )
        compressed_recovery = next(
            launch
            for launch in compressed.draft_launches
            if launch.kind is DraftJobKind.SPECULATIVE
        )
        self.assertLess(
            compressed_recovery.end_ms - compressed_recovery.start_ms,
            full_recovery.end_ms - full_recovery.start_ms,
        )
        self.assertLess(compressed.metrics.makespan_ms, full.metrics.makespan_ms)

    def test_adversarial_fixed_threshold_can_choose_expensive_wide_padding(self) -> None:
        trace = PreRealizedTrace(
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
        costs = _linear_costs(
            target_slot_ms=0.1,
            draft_context_ms=0.0,
            draft_token_ms=0.0,
        )
        forced_parallel = _run(
            trace,
            SPECTREHybridScheduler(
                SpectreCalibration(10.0, 1.0, 1.0),
                context_compression_factor=1.0,
            ),
            costs=costs,
            batch=1,
        )
        forced_ordinary = _run(
            trace,
            SPECTREHybridScheduler(
                SpectreCalibration(1.0, 10.0, 1.0),
                context_compression_factor=1.0,
            ),
            costs=costs,
            batch=1,
        )
        assert_semantic_equivalence(forced_parallel, forced_ordinary)
        self.assertGreater(forced_parallel.metrics.padded_slots, 0)
        self.assertGreater(
            forced_parallel.metrics.verifier_slots,
            forced_ordinary.metrics.verifier_slots,
        )
        self.assertGreater(
            forced_parallel.metrics.makespan_ms,
            forced_ordinary.metrics.makespan_ms,
        )


class EXSpecAbstractionTests(unittest.TestCase):
    def test_sliding_pool_groups_known_post_acceptance_lengths(self) -> None:
        trace = PreRealizedTrace(
            (
                RealizedRequest(
                    "a",
                    0.0,
                    100.0,
                    4,
                    (
                        _step(2, 0, (1,)),
                        _step(2, 1, (2, 3)),
                    ),
                ),
                RealizedRequest(
                    "b",
                    0.0,
                    100.0,
                    4,
                    (
                        _step(2, 1, (4, 5)),
                        _step(2, 0, (6,)),
                    ),
                ),
                RealizedRequest("match-a", 0.01, 100.0, 5, (_step(2, 0, (7,)),)),
                RealizedRequest("match-b", 0.01, 100.0, 6, (_step(2, 0, (8,)),)),
            )
        )
        result = _run(
            trace,
            EXSpecSlidingPoolScheduler(window_size=8),
            costs=_linear_costs(),
            batch=2,
        )
        first = result.target_launches[0]
        self.assertEqual(set(first.request_ids), {"a", "b"})
        later_pairs = [
            set(launch.request_ids)
            for launch in result.target_launches[1:]
            if len(launch.request_ids) == 2
        ]
        self.assertIn({"a", "match-a"}, later_pairs)
        self.assertIn({"b", "match-b"}, later_pairs)
        self.assertEqual(result.metrics.realignment_fallbacks, 0)

    def test_diverse_window_falls_back_to_correct_realign(self) -> None:
        trace = PreRealizedTrace(
            tuple(
                RealizedRequest(
                    f"r{index}",
                    0.0,
                    100.0,
                    index,
                    (_step(2, 0, (index,)),),
                )
                for index in range(4)
            )
        )
        result = _run(
            trace,
            EXSpecSlidingPoolScheduler(window_size=4),
            costs=_linear_costs(),
            batch=4,
        )
        self.assertEqual(result.metrics.realignment_fallbacks, 1)
        self.assertGreater(result.metrics.realignment_ms, 0.0)
        self.assertEqual(result.semantic_signature, trace.semantic_signature)

    def test_adversarial_grouping_can_starve_an_old_unique_length(self) -> None:
        pair_steps = tuple(_step(1, 0, (index,)) for index in range(8))
        trace = PreRealizedTrace(
            (
                RealizedRequest("unique", 0.0, 100.0, 99, (_step(1, 0, (99,)),)),
                RealizedRequest("pair-a", 0.0, 100.0, 0, pair_steps),
                RealizedRequest("pair-b", 0.0, 100.0, 0, pair_steps),
            ),
            name="exspec-starvation-counterexample",
        )
        costs = _linear_costs(starvation_ms=0.5)
        exspec = _run(
            trace,
            EXSpecSlidingPoolScheduler(window_size=3),
            costs=costs,
            batch=2,
        )
        fifo = _run(trace, FIFOScheduler(), costs=costs, batch=2)
        assert_semantic_equivalence(exspec, fifo)
        unique_exspec = next(
            request for request in exspec.requests if request.request_id == "unique"
        )
        unique_fifo = next(request for request in fifo.requests if request.request_id == "unique")
        self.assertGreater(
            unique_exspec.max_ready_wait_ms,
            unique_fifo.max_ready_wait_ms,
        )
        self.assertGreater(exspec.metrics.starved_requests, fifo.metrics.starved_requests)
        self.assertGreater(exspec.metrics.mean_real_batch, 1.0)


class MyopicSlackAbstractionTests(unittest.TestCase):
    def test_coalescing_and_aging_are_visible_in_wait_metrics(self) -> None:
        trace = PreRealizedTrace(
            (
                RealizedRequest("old", 0.0, 100.0, 0, (_step(1, 0, (1,)),)),
                RealizedRequest("new", 0.5, 2.0, 0, (_step(1, 0, (2,)),)),
            )
        )
        scheduler = MyopicSlackScheduler(
            estimated_base_ms=0.1,
            estimated_slot_ms=0.0,
            aging_rate=2.0,
            starvation_bound_ms=10.0,
            max_coalesce_ms=1.0,
        )
        result = _run(trace, scheduler, costs=_linear_costs(), batch=2)
        self.assertEqual(set(result.target_launches[0].request_ids), {"old", "new"})
        self.assertGreater(result.metrics.total_ready_wait_ms, 0.0)
        self.assertEqual(result.metrics.mean_real_batch, 2.0)

    def test_adversarial_aging_escape_can_block_a_new_tight_narrow_row(self) -> None:
        trace = PreRealizedTrace(
            (
                RealizedRequest("blocker", 0.0, 0.15, 0, (_step(1, 0, (1,)),)),
                RealizedRequest("old-wide", 0.0, 100.0, 0, (_step(50, 0, (2,)),)),
                RealizedRequest("new-tight", 0.19, 0.7, 0, (_step(1, 0, (3,)),)),
            ),
            name="aging-counterexample",
        )
        costs = _linear_costs(target_slot_ms=0.1)
        aging_escape = _run(
            trace,
            MyopicSlackScheduler(
                estimated_base_ms=0.1,
                estimated_slot_ms=0.1,
                aging_rate=1.0,
                starvation_bound_ms=0.15,
            ),
            costs=costs,
            batch=1,
        )
        slack_only = _run(
            trace,
            MyopicSlackScheduler(
                estimated_base_ms=0.1,
                estimated_slot_ms=0.1,
                aging_rate=0.0,
                starvation_bound_ms=1_000.0,
            ),
            costs=costs,
            batch=1,
        )
        assert_semantic_equivalence(aging_escape, slack_only)
        self.assertEqual(aging_escape.target_launches[1].request_ids, ("old-wide",))
        self.assertEqual(slack_only.target_launches[1].request_ids, ("new-tight",))
        tight_aging = next(
            request for request in aging_escape.requests if request.request_id == "new-tight"
        )
        tight_slack = next(
            request for request in slack_only.requests if request.request_id == "new-tight"
        )
        self.assertTrue(tight_aging.deadline_missed)
        self.assertFalse(tight_slack.deadline_missed)

    def test_adversarial_myopic_coalesce_waits_without_a_future_arrival(self) -> None:
        trace = PreRealizedTrace(
            (RealizedRequest("solo", 0.0, 100.0, 0, (_step(1, 0, (1,)),)),),
            name="coalesce-counterexample",
        )
        delayed = _run(
            trace,
            MyopicSlackScheduler(max_coalesce_ms=5.0),
            costs=_linear_costs(),
            batch=4,
        )
        immediate = _run(
            trace,
            MyopicSlackScheduler(max_coalesce_ms=0.0),
            costs=_linear_costs(),
            batch=4,
        )
        assert_semantic_equivalence(delayed, immediate)
        self.assertEqual(delayed.metrics.mean_real_batch, immediate.metrics.mean_real_batch)
        self.assertGreater(delayed.metrics.makespan_ms, immediate.metrics.makespan_ms)
        self.assertGreater(delayed.metrics.total_ready_wait_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
