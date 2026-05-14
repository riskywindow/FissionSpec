from __future__ import annotations

import unittest
from collections import defaultdict

from fissionspec.metrics import counterfactual_metrics, summarize
from fissionspec.model import Outcome
from fissionspec.policies import (
    FissionSpecPolicy,
    FixedCoalescePolicy,
    ImmediateFissionPolicy,
    SaguaroBarrierPolicy,
    SPECTREPaddedPolicy,
)
from fissionspec.profiles import HardwareProfile, LatencyCurve
from fissionspec.simulator import SimulationError, simulate
from fissionspec.workload import RequestConfig, Workload


class TableRNG:
    def __init__(
        self,
        values: dict[tuple[str, int], float] | None = None,
        *,
        default: float = 0.0,
    ) -> None:
        self.values = values or {}
        self.default = default
        self.calls: list[tuple[str, int, str, int]] = []

    def uniform(
        self, request_id: str, round_id: int, stream: str, draw: int = 0
    ) -> float:
        self.calls.append((request_id, round_id, stream, draw))
        return self.values.get((request_id, round_id), self.default)


def test_profile(*, recovery_ms: float = 5.0) -> HardwareProfile:
    return HardwareProfile(
        target_curve=LatencyCurve(((1, 1.0), (2, 1.2), (4, 1.5))),
        draft_curve=LatencyCurve(((1, 0.1), (2, 0.15), (4, 0.2))),
        recovery_curve=LatencyCurve(
            ((1, recovery_ms), (2, recovery_ms + 0.1), (4, recovery_ms + 0.2))
        ),
        verifier_slot_ms=0.0,
        name="test",
    )


def request(
    request_id: str,
    *,
    arrival: float = 0.0,
    output: int = 5,
    speculation: int = 2,
    probability: float = 1.0,
    slo: float = 50.0,
) -> RequestConfig:
    return RequestConfig(
        request_id=request_id,
        arrival_ms=arrival,
        output_tokens=output,
        speculation_length=speculation,
        acceptance_probability=probability,
        tbt_slo_ms=slo,
    )


class SimulatorPolicyTests(unittest.TestCase):
    def test_saguaro_miss_routes_entire_surviving_cohort_to_barrier(self) -> None:
        workload = Workload((request("hit"), request("miss", probability=0.0)))
        result = simulate(
            workload,
            test_profile(),
            SaguaroBarrierPolicy(),
            TableRNG(),
            max_batch_size=2,
        )
        first_target = result.target_launches[0]
        self.assertEqual(dict(first_target.outcomes)["hit"], Outcome.HIT)
        self.assertEqual(dict(first_target.outcomes)["miss"], Outcome.MISS)
        first_draft = result.draft_launches[0]
        self.assertTrue(first_draft.precompute)
        barrier = next(launch for launch in result.draft_launches if launch.barrier)
        self.assertTrue(barrier.recovery)
        self.assertEqual(barrier.request_ids, ("miss",))
        self.assertEqual(barrier.barrier_victim_ids, ("hit",))
        hit_result = next(item for item in result.requests if item.request_id == "hit")
        self.assertGreater(hit_result.hit_externality_ms, 0.0)

    def test_immediate_fission_splits_hit_and_miss_draft_work(self) -> None:
        workload = Workload((request("hit"), request("miss", probability=0.0)))
        result = simulate(
            workload,
            test_profile(),
            ImmediateFissionPolicy(),
            TableRNG(),
            max_batch_size=2,
        )
        self.assertEqual(set(result.draft_launches[0].request_ids), {"hit", "miss"})
        self.assertTrue(result.draft_launches[0].precompute)
        miss_recovery = next(
            launch
            for launch in result.draft_launches
            if launch.recovery and launch.request_ids == ("miss",)
        )
        self.assertTrue(miss_recovery.recovery)
        self.assertTrue(all(not launch.padded_request_ids for launch in result.target_launches))
        hit_result = next(item for item in result.requests if item.request_id == "hit")
        self.assertEqual(hit_result.hit_externality_ms, 0.0)

    def test_saguaro_final_token_miss_still_holds_surviving_hits(self) -> None:
        workload = Workload(
            (
                request("hit", output=3),
                request("final-miss", output=1, probability=0.0),
            )
        )
        result = simulate(
            workload,
            test_profile(),
            SaguaroBarrierPolicy(),
            TableRNG(),
            max_batch_size=2,
        )
        barrier = next(launch for launch in result.draft_launches if launch.barrier)
        self.assertEqual(barrier.request_ids, ("final-miss",))
        self.assertEqual(barrier.barrier_victim_ids, ("hit",))
        hit_second_launch = next(
            launch
            for launch in result.target_launches[1:]
            if launch.request_ids == ("hit",)
        )
        self.assertGreaterEqual(hit_second_launch.start_ms, barrier.end_ms)

    def test_spectre_keeps_recovery_as_one_token_padded_row(self) -> None:
        workload = Workload((request("hit"), request("miss", probability=0.0)))
        result = simulate(
            workload,
            test_profile(recovery_ms=8.0),
            SPECTREPaddedPolicy(),
            TableRNG(),
            max_batch_size=2,
        )
        padded_launches = [
            launch for launch in result.target_launches if launch.padded_request_ids
        ]
        self.assertTrue(padded_launches)
        first_padded = padded_launches[0]
        self.assertEqual(first_padded.padded_request_ids, ("miss",))
        self.assertEqual(first_padded.padded_verifier_slots, 1)
        recovery = next(
            launch
            for launch in result.draft_launches
            if launch.recovery and launch.request_ids == ("miss",)
        )
        miss_result = next(item for item in result.requests if item.request_id == "miss")
        self.assertTrue(any(time < recovery.end_ms for time in miss_result.token_times_ms[1:]))
        self.assertGreater(summarize(result).padded_verifier_slots, 0)

    def test_immediate_miss_leaves_target_until_recovery(self) -> None:
        workload = Workload((request("hit"), request("miss", probability=0.0)))
        result = simulate(
            workload,
            test_profile(recovery_ms=8.0),
            ImmediateFissionPolicy(),
            TableRNG(),
            max_batch_size=2,
        )
        recovery = next(
            launch
            for launch in result.draft_launches
            if launch.recovery and launch.request_ids == ("miss",)
        )
        miss_targets = [
            launch
            for launch in result.target_launches
            if "miss" in launch.request_ids
        ]
        self.assertGreaterEqual(miss_targets[1].start_ms, recovery.end_ms)
        self.assertEqual(result.padded_verifier_slots, 0)

    def test_spectre_can_issue_one_version_guarded_pad_only_launch(self) -> None:
        workload = Workload((request("miss", output=3, probability=0.0),))
        result = simulate(
            workload,
            test_profile(recovery_ms=8.0),
            SPECTREPaddedPolicy(),
            TableRNG(),
            max_batch_size=1,
        )
        pad_only = [
            launch
            for launch in result.target_launches
            if not launch.request_ids and launch.padded_request_ids == ("miss",)
        ]
        self.assertEqual(len(pad_only), 1)
        # Its target-only token invalidates the already-running recovery, so a
        # versioned recovery reissue must appear before speculative dispatch.
        recoveries = [launch for launch in result.draft_launches if launch.recovery]
        self.assertGreaterEqual(len(recoveries), 2)

    def test_target_timeline_progresses_while_draft_recovery_is_busy(self) -> None:
        workload = Workload(
            (
                request("miss", output=2, probability=0.0),
                request("later", arrival=1.1, output=1, probability=1.0),
            )
        )
        result = simulate(
            workload,
            test_profile(recovery_ms=10.0),
            ImmediateFissionPolicy(),
            TableRNG(),
            max_batch_size=1,
        )
        recovery = next(launch for launch in result.draft_launches if launch.recovery)
        later_target = next(
            launch for launch in result.target_launches if launch.request_ids == ("later",)
        )
        self.assertLess(later_target.start_ms, recovery.end_ms)

    def test_ssd_precompute_overlaps_target_and_hit_consumes_cached_branch(self) -> None:
        workload = Workload((request("hit", output=5, probability=1.0),))
        result = simulate(
            workload,
            test_profile(),
            ImmediateFissionPolicy(),
            TableRNG(),
            max_batch_size=1,
        )
        first_target = result.target_launches[0]
        first_precompute = result.draft_launches[0]
        self.assertTrue(first_precompute.precompute)
        self.assertEqual(first_precompute.start_ms, first_target.start_ms)
        self.assertLess(first_precompute.end_ms, first_target.end_ms)
        self.assertEqual(result.target_launches[1].start_ms, first_target.end_ms)

    def test_hit_waits_when_precompute_is_slower_than_verification(self) -> None:
        profile = HardwareProfile(
            target_curve=LatencyCurve(((1, 1.0),)),
            draft_curve=LatencyCurve(((1, 5.0),)),
            recovery_curve=LatencyCurve(((1, 6.0),)),
            verifier_slot_ms=0.0,
            name="draft-bound",
        )
        result = simulate(
            Workload((request("hit", output=3, probability=1.0),)),
            profile,
            ImmediateFissionPolicy(),
            TableRNG(),
            max_batch_size=1,
        )
        self.assertEqual(result.target_launches[0].end_ms, 1.0)
        self.assertEqual(result.draft_launches[0].end_ms, 5.0)
        self.assertEqual(result.target_launches[1].start_ms, 5.0)

    def test_fixed_coalesce_combines_staggered_arrivals(self) -> None:
        workload = Workload(
            (
                request("a", output=1),
                request("b", arrival=0.5, output=1),
            )
        )
        profile = test_profile()
        fixed = simulate(
            workload,
            profile,
            FixedCoalescePolicy(coalesce_ms=1.0),
            TableRNG(),
            max_batch_size=4,
        )
        immediate = simulate(
            workload,
            profile,
            ImmediateFissionPolicy(),
            TableRNG(),
            max_batch_size=4,
        )
        self.assertEqual(fixed.target_launches[0].request_ids, ("a", "b"))
        self.assertEqual(len(fixed.target_launches), 1)
        self.assertEqual(len(immediate.target_launches), 2)

    def test_horizon_two_controller_coalesces_when_flow_cost_improves(self) -> None:
        workload = Workload(
            (request("a", output=1), request("b", arrival=0.1, output=1))
        )
        profile = HardwareProfile.linear(
            target_overhead_ms=10.0,
            target_per_row_ms=0.1,
            draft_overhead_ms=0.1,
            draft_per_row_ms=0.01,
            recovery_overhead_ms=1.0,
            recovery_per_row_ms=0.1,
            verifier_slot_ms=0.0,
        )
        result = simulate(
            workload,
            profile,
            FissionSpecPolicy(max_wait_ms=1.0),
            TableRNG(),
            max_batch_size=4,
        )
        self.assertEqual(len(result.target_launches), 1)
        self.assertEqual(result.target_launches[0].start_ms, 0.1)


class DeterminismAndMetricTests(unittest.TestCase):
    def test_policy_changes_do_not_change_rng_addressed_outcomes(self) -> None:
        workload = Workload(
            (
                request("a", output=9, probability=0.5),
                request("b", arrival=0.2, output=9, probability=0.5),
            )
        )
        values = {
            ("a", 0): 0.1,
            ("a", 1): 0.9,
            ("a", 2): 0.2,
            ("a", 3): 0.8,
            ("b", 0): 0.7,
            ("b", 1): 0.3,
            ("b", 2): 0.6,
            ("b", 3): 0.1,
        }
        rng_immediate = TableRNG(values)
        rng_fixed = TableRNG(values)
        immediate = simulate(
            workload,
            test_profile(),
            ImmediateFissionPolicy(),
            rng_immediate,
            max_batch_size=2,
        )
        fixed = simulate(
            workload,
            test_profile(),
            FixedCoalescePolicy(coalesce_ms=0.5),
            rng_fixed,
            max_batch_size=2,
        )

        def traces(result: object) -> dict[str, list[Outcome]]:
            by_request: dict[str, list[Outcome]] = defaultdict(list)
            for launch in result.target_launches:  # type: ignore[attr-defined]
                for request_id, outcome in launch.outcomes:
                    by_request[request_id].append(outcome)
            return by_request

        self.assertEqual(traces(immediate), traces(fixed))
        self.assertTrue(
            all(stream == "acceptance" and draw == 0 for _, _, stream, draw in rng_fixed.calls)
        )

    def test_metrics_and_counterfactual_are_complete(self) -> None:
        workload = Workload((request("a", output=5, slo=0.5), request("b", output=5)))
        profile = test_profile()
        baseline = simulate(
            workload,
            profile,
            SaguaroBarrierPolicy(),
            TableRNG({("b", 0): 0.99}),
            max_batch_size=2,
        )
        candidate = simulate(
            workload,
            profile,
            ImmediateFissionPolicy(),
            TableRNG({("b", 0): 0.99}),
            max_batch_size=2,
        )
        metrics = summarize(candidate)
        self.assertGreater(metrics.throughput_tokens_per_s, 0.0)
        self.assertGreaterEqual(metrics.p99_tbt_ms, metrics.p50_tbt_ms)
        self.assertGreaterEqual(metrics.slo_attainment, 0.0)
        self.assertLessEqual(metrics.slo_attainment, 1.0)
        self.assertGreater(metrics.target_launches, 0)
        self.assertGreater(metrics.mean_batch, 0.0)
        comparison = counterfactual_metrics(candidate, baseline)
        self.assertEqual(comparison.candidate.policy_name, "immediate-fission")
        self.assertEqual(comparison.baseline.policy_name, "saguaro-barrier")

    def test_invalid_rng_value_fails_loudly(self) -> None:
        workload = Workload((request("a", output=1),))
        with self.assertRaises(SimulationError):
            simulate(
                workload,
                test_profile(),
                ImmediateFissionPolicy(),
                TableRNG(default=1.0),
            )


if __name__ == "__main__":
    unittest.main()
