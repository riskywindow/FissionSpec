from __future__ import annotations

import hashlib
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
from fissionspec.rng import CounterRNG
from fissionspec.simulator import SimulationError, simulate
from fissionspec.workload import RequestConfig, Workload


class TableRNG:
    def __init__(
        self,
        values: dict[tuple[str, int, str, int], float] | None = None,
        *,
        default: float = 0.0,
    ) -> None:
        self.values = values or {}
        self.default = default
        self.calls: list[tuple[str, int, str, int]] = []

    @property
    def provenance(self) -> str:
        payload = repr((sorted(self.values.items()), self.default)).encode()
        return f"table-rng-v1:{hashlib.sha256(payload).hexdigest()}"

    def uniform(self, request_id: str, round_id: int, stream: str, draw: int = 0) -> float:
        self.calls.append((request_id, round_id, stream, draw))
        return self.values.get((request_id, round_id, stream, draw), self.default)


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
    cache_hit: float = 1.0,
    token_acceptance: float = 1.0,
    slo: float = 50.0,
) -> RequestConfig:
    return RequestConfig(
        request_id=request_id,
        arrival_ms=arrival,
        output_tokens=output,
        speculation_length=speculation,
        cache_hit_probability=cache_hit,
        token_acceptance_probability=token_acceptance,
        tbt_slo_ms=slo,
    )


class SimulatorPolicyTests(unittest.TestCase):
    def test_workload_rejects_bool_time_and_non_string_identity(self) -> None:
        with self.assertRaisesRegex(ValueError, "arrival_ms"):
            request("bad-time", arrival=True)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "request_id"):
            request(7)  # type: ignore[arg-type]

    def test_cache_outcome_is_independent_of_token_productivity(self) -> None:
        workload = Workload(
            (
                request(
                    "forced-hit",
                    output=8,
                    speculation=4,
                    cache_hit=1.0,
                    token_acceptance=0.5,
                ),
                request(
                    "forced-miss",
                    output=8,
                    speculation=4,
                    cache_hit=0.0,
                    token_acceptance=0.5,
                ),
            )
        )
        draws = {
            (request_id, 0, "token-acceptance", 0): 0.1
            for request_id in ("forced-hit", "forced-miss")
        }
        draws.update(
            {
                (request_id, 0, "token-acceptance", 1): 0.2
                for request_id in ("forced-hit", "forced-miss")
            }
        )
        draws.update(
            {
                (request_id, 0, "token-acceptance", 2): 0.9
                for request_id in ("forced-hit", "forced-miss")
            }
        )
        first = simulate(
            workload,
            test_profile(),
            ImmediateFissionPolicy(),
            TableRNG(draws),
            max_batch_size=2,
        ).target_launches[0]
        self.assertEqual(dict(first.accepted_tokens), {"forced-hit": 2, "forced-miss": 2})
        self.assertEqual(dict(first.productive_tokens), {"forced-hit": 3, "forced-miss": 3})
        self.assertEqual(dict(first.outcomes)["forced-hit"], Outcome.HIT)
        self.assertEqual(dict(first.outcomes)["forced-miss"], Outcome.MISS)

    def test_token_productivity_boundaries_are_one_to_width(self) -> None:
        workload = Workload(
            (
                request(
                    "reject",
                    output=4,
                    speculation=4,
                    token_acceptance=0.0,
                ),
                request(
                    "accept",
                    output=4,
                    speculation=4,
                    token_acceptance=1.0,
                ),
            )
        )
        rng = TableRNG()
        first = simulate(
            workload,
            test_profile(),
            ImmediateFissionPolicy(),
            rng,
            max_batch_size=2,
        ).target_launches[0]
        self.assertEqual(dict(first.accepted_tokens), {"accept": 3, "reject": 0})
        self.assertEqual(dict(first.productive_tokens), {"accept": 4, "reject": 1})
        accept_draws = [
            draw
            for request_id, round_id, stream, draw in rng.calls
            if request_id == "accept" and round_id == 0 and stream == "token-acceptance"
        ]
        self.assertEqual(accept_draws, [0, 1, 2])

    def test_terminal_round_does_not_invent_next_continuation_lookup(self) -> None:
        rng = TableRNG()
        result = simulate(
            Workload((request("terminal", output=1, speculation=4),)),
            test_profile(),
            ImmediateFissionPolicy(),
            rng,
            max_batch_size=1,
        )
        self.assertEqual(result.target_launches[0].outcomes, (("terminal", Outcome.TERMINAL),))
        self.assertFalse(any(stream == "cache-hit" for _, _, stream, _ in rng.calls))
        metrics = summarize(result)
        self.assertEqual(metrics.cache_hits, 0)
        self.assertEqual(metrics.cache_misses, 0)
        self.assertEqual(metrics.verifier_rounds, 1)
        self.assertEqual(metrics.mean_verifier_tokens_per_round, 1.0)
        self.assertEqual(metrics.request_tbt_slo_attainment, 1.0)
        self.assertGreater(metrics.tbt_request_goodput_tokens_per_s, 0.0)

    def test_saguaro_miss_routes_entire_surviving_cohort_to_barrier(self) -> None:
        workload = Workload((request("hit", output=3), request("miss", cache_hit=0.0)))
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
        local_eligibility = max(first_target.end_ms, first_draft.end_ms)
        self.assertAlmostEqual(
            hit_result.direct_hit_delay_ms,
            barrier.end_ms - local_eligibility,
        )

    def test_immediate_fission_splits_hit_and_miss_draft_work(self) -> None:
        workload = Workload((request("hit"), request("miss", cache_hit=0.0)))
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
        self.assertEqual(hit_result.direct_hit_delay_ms, 0.0)

    def test_saguaro_completed_miss_does_not_hold_surviving_hits(self) -> None:
        workload = Workload(
            (
                request("hit", output=3),
                request("final-miss", output=1, cache_hit=0.0),
            )
        )
        result = simulate(
            workload,
            test_profile(),
            SaguaroBarrierPolicy(),
            TableRNG(),
            max_batch_size=2,
        )
        self.assertFalse(any(launch.barrier for launch in result.draft_launches))
        hit_second_launch = next(
            launch for launch in result.target_launches[1:] if launch.request_ids == ("hit",)
        )
        first_target = result.target_launches[0]
        first_precompute = next(launch for launch in result.draft_launches if launch.precompute)
        self.assertEqual(
            hit_second_launch.start_ms,
            max(first_target.end_ms, first_precompute.end_ms),
        )

    def test_spectre_keeps_recovery_as_one_token_padded_row(self) -> None:
        workload = Workload((request("hit"), request("miss", cache_hit=0.0)))
        result = simulate(
            workload,
            test_profile(recovery_ms=8.0),
            SPECTREPaddedPolicy(),
            TableRNG(),
            max_batch_size=2,
        )
        padded_launches = [launch for launch in result.target_launches if launch.padded_request_ids]
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
        workload = Workload((request("hit"), request("miss", cache_hit=0.0)))
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
        miss_targets = [launch for launch in result.target_launches if "miss" in launch.request_ids]
        self.assertGreaterEqual(miss_targets[1].start_ms, recovery.end_ms)
        self.assertEqual(result.padded_verifier_slots, 0)

    def test_spectre_can_issue_one_version_guarded_pad_only_launch(self) -> None:
        workload = Workload(
            (
                request(
                    "miss",
                    output=3,
                    cache_hit=0.0,
                    token_acceptance=0.0,
                ),
            )
        )
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

    def test_spectre_padding_never_exceeds_batch_capacity(self) -> None:
        workload = Workload(
            tuple(
                request(
                    f"miss-{index}",
                    output=5,
                    cache_hit=0.0,
                    token_acceptance=0.0,
                )
                for index in range(8)
            )
        )
        result = simulate(
            workload,
            test_profile(recovery_ms=4.0),
            SPECTREPaddedPolicy(),
            TableRNG(),
            max_batch_size=3,
        )
        self.assertEqual(result.policy_name, "spectre-parallel-padded")
        self.assertTrue(all(launch.effective_batch_size <= 3 for launch in result.target_launches))

    def test_spectre_padding_never_displaces_productive_ready_rows(self) -> None:
        workload = Workload(
            (
                request(
                    "miss",
                    output=4,
                    cache_hit=0.0,
                    token_acceptance=0.0,
                ),
                request("later-a", arrival=0.9, output=1),
                request("later-b", arrival=0.9, output=1),
            )
        )
        result = simulate(
            workload,
            test_profile(recovery_ms=8.0),
            SPECTREPaddedPolicy(),
            TableRNG(),
            max_batch_size=2,
        )
        launch = next(item for item in result.target_launches if "later-a" in item.request_ids)
        self.assertEqual(launch.request_ids, ("later-a", "later-b"))
        self.assertEqual(launch.padded_request_ids, ())

    def test_final_real_row_caps_slots_and_skips_known_useless_precompute(self) -> None:
        result = simulate(
            Workload((request("one", output=1, speculation=8),)),
            HardwareProfile.linear(verifier_slot_ms=0.25),
            ImmediateFissionPolicy(),
            TableRNG(),
            max_batch_size=1,
        )
        self.assertEqual(result.target_launches[0].verifier_slots, 1)
        self.assertFalse(any(launch.precompute for launch in result.draft_launches))

    def test_deadline_admission_prevents_fifo_head_of_line_inversion(self) -> None:
        loose = RequestConfig(
            request_id="loose-first",
            arrival_ms=0.0,
            output_tokens=1,
            speculation_length=1,
            deadline_ms=100.0,
        )
        urgent = RequestConfig(
            request_id="urgent-second",
            arrival_ms=0.0,
            output_tokens=1,
            speculation_length=1,
            deadline_ms=1.1,
        )
        result = simulate(
            Workload((loose, urgent)),
            test_profile(),
            ImmediateFissionPolicy(),
            TableRNG(),
            max_batch_size=1,
        )
        self.assertEqual(result.target_launches[0].request_ids, ("urgent-second",))
        urgent_result = next(item for item in result.requests if item.request_id == "urgent-second")
        self.assertLessEqual(urgent_result.completion_ms, urgent.deadline_ms)

    def test_target_timeline_progresses_while_draft_recovery_is_busy(self) -> None:
        workload = Workload(
            (
                request(
                    "miss",
                    output=2,
                    cache_hit=0.0,
                    token_acceptance=0.0,
                ),
                request("later", arrival=1.1, output=1, cache_hit=1.0),
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
        workload = Workload((request("hit", output=5, cache_hit=1.0),))
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
            Workload((request("hit", output=3, cache_hit=1.0),)),
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

    def test_horizon_two_does_not_forecast_external_arrivals(self) -> None:
        workload = Workload((request("a", output=1), request("b", arrival=0.1, output=1)))
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
        self.assertEqual(len(result.target_launches), 2)
        self.assertEqual(result.target_launches[0].start_ms, 0.0)
        self.assertEqual(result.target_launches[0].request_ids, ("a",))

    def test_horizon_two_coalesces_with_known_recovery_eta(self) -> None:
        workload = Workload(
            (
                request(
                    "hit",
                    output=2,
                    speculation=1,
                    cache_hit=1.0,
                    token_acceptance=0.0,
                ),
                request(
                    "miss",
                    output=2,
                    speculation=1,
                    cache_hit=0.0,
                    token_acceptance=0.0,
                ),
            )
        )
        profile = HardwareProfile.linear(
            target_overhead_ms=10.0,
            target_per_row_ms=0.1,
            draft_overhead_ms=0.01,
            draft_per_row_ms=0.01,
            recovery_overhead_ms=0.1,
            recovery_per_row_ms=0.01,
            verifier_slot_ms=0.0,
        )
        result = simulate(
            workload,
            profile,
            FissionSpecPolicy(max_wait_ms=1.0),
            TableRNG(),
            max_batch_size=4,
        )
        recovery = next(launch for launch in result.draft_launches if launch.recovery)
        fused = result.target_launches[1]
        self.assertEqual(fused.start_ms, recovery.end_ms)
        self.assertEqual(fused.request_ids, ("hit", "miss"))

    def test_horizon_two_forecast_matches_edf_heterogeneous_overflow(self) -> None:
        workload = Workload(
            (
                RequestConfig(
                    "a-small",
                    output_tokens=3,
                    speculation_length=1,
                    cache_hit_probability=0.0,
                    token_acceptance_probability=0.0,
                    deadline_ms=50.0,
                ),
                RequestConfig(
                    "b-medium",
                    output_tokens=9,
                    speculation_length=8,
                    cache_hit_probability=0.0,
                    token_acceptance_probability=0.0,
                    deadline_ms=50.0,
                ),
                RequestConfig(
                    "c-large",
                    output_tokens=17,
                    speculation_length=16,
                    cache_hit_probability=0.0,
                    token_acceptance_probability=0.0,
                    deadline_ms=50.0,
                ),
                RequestConfig(
                    "loose-current",
                    arrival_ms=0.381,
                    output_tokens=1,
                    speculation_length=1,
                    deadline_ms=100.0,
                ),
            )
        )
        profile = HardwareProfile.linear(
            target_overhead_ms=0.1,
            target_per_row_ms=0.01,
            draft_overhead_ms=0.0,
            draft_per_row_ms=0.001,
            recovery_overhead_ms=0.001,
            recovery_per_row_ms=0.001,
            verifier_slot_ms=0.01,
            name="edf-overflow-regression",
        )
        result = simulate(
            workload,
            profile,
            FissionSpecPolicy(max_wait_ms=2.0),
            CounterRNG(0),
            max_batch_size=3,
        )
        recovery = next(launch for launch in result.draft_launches if launch.recovery)
        current_launch = result.target_launches[1]
        # At recovery, EDF would place all three earlier-deadline rows ahead of
        # loose-current, including their heterogeneous widths (1, 8, 16). The
        # exact forecast therefore dispatches loose-current before that wake.
        self.assertEqual(current_launch.request_ids, ("loose-current",))
        self.assertEqual(current_launch.start_ms, 0.381)
        self.assertLess(current_launch.start_ms, recovery.end_ms)

    def test_horizon_two_protects_rolling_next_token_slo(self) -> None:
        workload = Workload(
            (
                request(
                    "urgent-hit",
                    output=100,
                    speculation=1,
                    cache_hit=1.0,
                    token_acceptance=0.0,
                    slo=1.2,
                ),
                request(
                    "recovering-miss",
                    output=2,
                    speculation=1,
                    cache_hit=0.0,
                    token_acceptance=0.0,
                ),
            )
        )
        profile = HardwareProfile.linear(
            target_overhead_ms=1.0,
            target_per_row_ms=0.01,
            draft_overhead_ms=0.01,
            draft_per_row_ms=0.01,
            recovery_overhead_ms=0.5,
            recovery_per_row_ms=0.01,
            verifier_slot_ms=0.0,
        )
        result = simulate(
            workload,
            profile,
            FissionSpecPolicy(max_wait_ms=2.0),
            TableRNG(),
            max_batch_size=4,
        )
        recovery = next(launch for launch in result.draft_launches if launch.recovery)
        first = result.target_launches[0]
        second = result.target_launches[1]
        # After the first verifier emits urgent-hit's first token, launching it
        # alone meets the rolling 1.2 ms TBT bound. Waiting for the known miss
        # recovery and fusing would not, despite a loose 120 ms final budget.
        self.assertLess(second.start_ms, recovery.end_ms)
        self.assertEqual(second.start_ms, first.end_ms)
        self.assertEqual(second.request_ids, ("urgent-hit",))


class DeterminismAndMetricTests(unittest.TestCase):
    def test_policy_changes_do_not_change_rng_addressed_outcomes(self) -> None:
        workload = Workload(
            (
                request("a", output=9, cache_hit=0.5, token_acceptance=0.5),
                request(
                    "b",
                    arrival=0.2,
                    output=9,
                    cache_hit=0.5,
                    token_acceptance=0.5,
                ),
            )
        )
        values = {
            ("a", 0, "cache-hit", 0): 0.1,
            ("a", 1, "cache-hit", 0): 0.9,
            ("a", 2, "cache-hit", 0): 0.2,
            ("a", 3, "cache-hit", 0): 0.8,
            ("b", 0, "cache-hit", 0): 0.7,
            ("b", 1, "cache-hit", 0): 0.3,
            ("b", 2, "cache-hit", 0): 0.6,
            ("b", 3, "cache-hit", 0): 0.1,
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
            all(stream in {"cache-hit", "token-acceptance"} for _, _, stream, _ in rng_fixed.calls)
        )

    def test_metrics_and_counterfactual_are_complete(self) -> None:
        workload = Workload((request("a", output=5, slo=0.5), request("b", output=5)))
        profile = test_profile()
        baseline = simulate(
            workload,
            profile,
            SaguaroBarrierPolicy(),
            TableRNG({("b", 0, "cache-hit", 0): 0.99}),
            max_batch_size=2,
        )
        candidate = simulate(
            workload,
            profile,
            ImmediateFissionPolicy(),
            TableRNG({("b", 0, "cache-hit", 0): 0.99}),
            max_batch_size=2,
        )
        metrics = summarize(candidate)
        self.assertGreater(metrics.throughput_tokens_per_s, 0.0)
        self.assertGreaterEqual(metrics.p99_tbt_ms, metrics.p50_tbt_ms)
        self.assertGreaterEqual(metrics.token_gap_slo_attainment, 0.0)
        self.assertLessEqual(metrics.token_gap_slo_attainment, 1.0)
        self.assertGreaterEqual(metrics.request_tbt_slo_attainment, 0.0)
        self.assertLessEqual(metrics.request_tbt_slo_attainment, 1.0)
        self.assertGreaterEqual(metrics.tbt_request_goodput_tokens_per_s, 0.0)
        self.assertGreater(metrics.target_launches, 0)
        self.assertGreater(metrics.mean_batch, 0.0)
        self.assertLess(
            metrics.cache_hits + metrics.cache_misses,
            metrics.verifier_rounds,
        )
        self.assertEqual(
            metrics.verifier_rounds,
            sum(launch.real_batch_size for launch in candidate.target_launches),
        )
        self.assertGreater(metrics.verifier_emitted_tokens, 0)
        self.assertGreater(metrics.mean_verifier_tokens_per_round, 0.0)
        comparison = counterfactual_metrics(candidate, baseline)
        self.assertEqual(comparison.candidate.policy_name, "immediate-fission")
        self.assertEqual(comparison.baseline.policy_name, "saguaro-barrier")

    def test_counterfactual_pairing_rejects_config_and_hardware_mismatch(self) -> None:
        baseline_workload = Workload((request("a", output=2),), name="paired")
        changed_workload = Workload(
            (request("a", output=3),),
            name="paired",
        )
        baseline = simulate(
            baseline_workload,
            test_profile(),
            ImmediateFissionPolicy(),
            TableRNG(),
        )
        changed = simulate(
            changed_workload,
            test_profile(),
            ImmediateFissionPolicy(),
            TableRNG(),
        )
        with self.assertRaisesRegex(ValueError, "workload configs"):
            counterfactual_metrics(changed, baseline)

        changed_hardware = simulate(
            baseline_workload,
            HardwareProfile.linear(name="different"),
            ImmediateFissionPolicy(),
            TableRNG(),
        )
        with self.assertRaisesRegex(ValueError, "hardware profile"):
            counterfactual_metrics(changed_hardware, baseline)

        changed_rng = simulate(
            baseline_workload,
            test_profile(),
            ImmediateFissionPolicy(),
            TableRNG(default=0.25),
        )
        with self.assertRaisesRegex(ValueError, "RNG provenance"):
            counterfactual_metrics(changed_rng, baseline)

    def test_randomized_policy_matrix_is_live_and_capacity_safe(self) -> None:
        policies = (
            SaguaroBarrierPolicy(),
            SPECTREPaddedPolicy(),
            ImmediateFissionPolicy(),
            FixedCoalescePolicy(coalesce_ms=0.3),
            FissionSpecPolicy(max_wait_ms=0.6),
        )
        workload = Workload.homogeneous(
            7,
            arrival_interval_ms=0.17,
            output_tokens=13,
            speculation_length=4,
            cache_hit_probability=0.55,
            token_acceptance_probability=0.65,
        )
        for seed in range(8):
            for policy in policies:
                with self.subTest(seed=seed, policy=policy.name):
                    result = simulate(
                        workload,
                        test_profile(recovery_ms=1.7),
                        policy,
                        CounterRNG(seed),
                        max_batch_size=3,
                        max_events=50_000,
                    )
                    self.assertEqual(len(result.requests), len(workload))
                    self.assertTrue(
                        all(launch.effective_batch_size <= 3 for launch in result.target_launches)
                    )

    def test_invalid_rng_value_fails_loudly(self) -> None:
        workload = Workload((request("a", output=2, speculation=2),))
        with self.assertRaises(SimulationError):
            simulate(
                workload,
                test_profile(),
                ImmediateFissionPolicy(),
                TableRNG(default=1.0),
            )


if __name__ == "__main__":
    unittest.main()
