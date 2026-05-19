from __future__ import annotations

import unittest

from fissionspec.policies import (
    DispatchContext,
    FissionSpecPolicy,
    FixedCoalescePolicy,
    ImmediateFissionPolicy,
    SaguaroBarrierPolicy,
    SPECTREPaddedPolicy,
    policy_from_name,
)
from fissionspec.profiles import HardwareProfile


def context(
    profile: HardwareProfile,
    *,
    now: float = 0.0,
    ready: int = 1,
    capacity: int = 8,
    next_time: float | None = 0.1,
    next_count: int = 1,
    deadline: float = 1_000.0,
    oldest: float | None = None,
    future_deadline: float | None = 1_000.0,
    next_slots: int | None = 1,
    row_deadlines: tuple[float, ...] | None = None,
    future_row_deadlines: tuple[float, ...] | None = None,
) -> DispatchContext:
    selected_rows = min(ready, capacity)
    effective_next_count = 0 if next_time is None else next_count
    return DispatchContext(
        now_ms=now,
        ready_count=ready,
        capacity=capacity,
        oldest_ready_ms=now if oldest is None else oldest,
        earliest_deadline_ms=deadline,
        row_slots=(1,) * selected_rows,
        row_deadlines_ms=((deadline,) * selected_rows if row_deadlines is None else row_deadlines),
        profile=profile,
        next_ready_time_ms=next_time,
        next_ready_count=effective_next_count,
        earliest_future_deadline_ms=future_deadline,
        future_row_slots=(1 if next_slots is None else next_slots,) * effective_next_count,
        future_row_deadlines_ms=(
            ((1_000.0 if future_deadline is None else future_deadline),) * effective_next_count
            if future_row_deadlines is None
            else future_row_deadlines
        ),
    )


class PolicySemanticsTests(unittest.TestCase):
    def test_policy_configuration_rejects_bool_as_duration(self) -> None:
        with self.assertRaises(ValueError):
            FixedCoalescePolicy(coalesce_ms=True)
        with self.assertRaises(ValueError):
            FissionSpecPolicy(max_wait_ms=True)

    def test_policy_flags_encode_cohort_semantics(self) -> None:
        self.assertTrue(SaguaroBarrierPolicy().barrier_on_miss)
        self.assertFalse(SaguaroBarrierPolicy().pad_recovering_misses)
        self.assertTrue(SPECTREPaddedPolicy().pad_recovering_misses)
        self.assertFalse(ImmediateFissionPolicy().barrier_on_miss)

    def test_policy_parser_supports_stable_and_short_names(self) -> None:
        self.assertIsInstance(policy_from_name("saguaro"), SaguaroBarrierPolicy)
        self.assertIsInstance(policy_from_name("spectre-padded"), SPECTREPaddedPolicy)
        self.assertEqual(policy_from_name("spectre").name, "spectre-parallel-padded")
        self.assertIsInstance(policy_from_name("fission"), ImmediateFissionPolicy)
        self.assertIsInstance(policy_from_name("horizon_2"), FissionSpecPolicy)
        with self.assertRaises(ValueError):
            policy_from_name("not-a-policy")


class FixedCoalesceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = HardwareProfile.linear(verifier_slot_ms=0.0)

    def test_waits_from_oldest_ready_time(self) -> None:
        policy = FixedCoalescePolicy(coalesce_ms=2.0)
        decision = policy.dispatch_at(context(self.profile, now=3.0, next_time=None))
        self.assertEqual(decision, 5.0)

    def test_full_batch_dispatches_immediately(self) -> None:
        policy = FixedCoalescePolicy(coalesce_ms=2.0)
        decision = policy.dispatch_at(
            context(self.profile, now=3.0, ready=8, capacity=8, next_time=None)
        )
        self.assertEqual(decision, 3.0)

    def test_deadline_caps_the_batching_window(self) -> None:
        policy = FixedCoalescePolicy(coalesce_ms=100.0)
        ctx = context(self.profile, now=4.0, deadline=5.3, next_time=None)
        self.assertAlmostEqual(
            policy.dispatch_at(ctx),
            5.3 - self.profile.target_latency_ms(1, 1),
        )


class HorizonTwoTests(unittest.TestCase):
    def test_coalesces_when_launch_overhead_dominates(self) -> None:
        profile = HardwareProfile.linear(
            target_overhead_ms=10.0,
            target_per_row_ms=1.0,
            verifier_slot_ms=0.0,
        )
        policy = FissionSpecPolicy(max_wait_ms=1.0)
        self.assertEqual(policy.dispatch_at(context(profile, next_time=0.1)), 0.1)

    def test_launches_now_when_per_row_cost_dominates(self) -> None:
        profile = HardwareProfile.linear(
            target_overhead_ms=0.0,
            target_per_row_ms=1.0,
            verifier_slot_ms=0.0,
        )
        policy = FissionSpecPolicy(max_wait_ms=1.0)
        self.assertEqual(policy.dispatch_at(context(profile, next_time=0.5)), 0.0)

    def test_documented_aggregate_flow_cost_boundary(self) -> None:
        profile = HardwareProfile.linear(
            target_overhead_ms=4.0,
            target_per_row_ms=0.5,
            verifier_slot_ms=0.0,
        )
        policy = FissionSpecPolicy(max_wait_ms=2.0)
        ctx = context(profile, ready=2, next_time=0.25, next_count=3)
        n, m, delta = 2, 3, 0.25
        latency_n = profile.target_latency_ms(n, n)
        latency_m = profile.target_latency_ms(m, m)
        latency_nm = profile.target_latency_ms(n + m, n + m)
        cost_now = n * latency_n + m * (max(latency_n, delta) + latency_m - delta)
        cost_wait = n * (delta + latency_nm) + m * latency_nm
        self.assertLess(cost_wait, cost_now)
        self.assertEqual(policy.dispatch_at(ctx), delta)

    def test_controller_prices_exact_packed_row_widths(self) -> None:
        profile = HardwareProfile.linear(
            target_overhead_ms=0.5,
            target_per_row_ms=0.01,
            verifier_slot_ms=0.5,
        )
        policy = FissionSpecPolicy(max_wait_ms=1.0)
        ctx = DispatchContext(
            now_ms=0.0,
            ready_count=2,
            capacity=3,
            oldest_ready_ms=0.0,
            earliest_deadline_ms=100.0,
            row_slots=(1, 2),
            row_deadlines_ms=(100.0, 100.0),
            profile=profile,
            next_ready_time_ms=0.05,
            next_ready_count=2,
            earliest_future_deadline_ms=100.0,
            future_row_slots=(1, 2),
            future_row_deadlines_ms=(100.0, 100.0),
        )
        # Exact packed widths choose fusion. Pricing both heterogeneous sets as
        # rows*max(width) would instead choose dispatch in this trace.
        self.assertEqual(policy.dispatch_at(ctx), 0.05)

    def test_wait_forecast_globally_merges_edf_rows_before_overflow(self) -> None:
        profile = HardwareProfile.linear(
            target_overhead_ms=0.1,
            target_per_row_ms=0.01,
            verifier_slot_ms=0.01,
        )
        policy = FissionSpecPolicy(max_wait_ms=2.0)
        ctx = DispatchContext(
            now_ms=0.0,
            ready_count=1,
            capacity=3,
            oldest_ready_ms=0.0,
            earliest_deadline_ms=100.0,
            row_slots=(1,),
            row_deadlines_ms=(100.0,),
            profile=profile,
            next_ready_time_ms=0.01,
            next_ready_count=3,
            earliest_future_deadline_ms=50.0,
            future_row_slots=(1, 8, 16),
            future_row_deadlines_ms=(50.0, 50.0, 50.0),
        )
        # At the wake all three urgent future rows precede the loose current
        # row, so the width-1 current row overflows. A current-first forecast
        # would put widths (1, 1, 8) together and incorrectly choose to wait.
        self.assertEqual(policy.dispatch_at(ctx), 0.0)

    def test_max_wait_deadline_and_capacity_force_launch(self) -> None:
        profile = HardwareProfile.linear(target_overhead_ms=10.0, target_per_row_ms=0.1)
        policy = FissionSpecPolicy(max_wait_ms=1.0)
        self.assertEqual(policy.dispatch_at(context(profile, next_time=2.0)), 0.0)
        self.assertEqual(
            policy.dispatch_at(context(profile, ready=8, capacity=8)),
            0.0,
        )

    def test_cumulative_wait_is_anchored_to_oldest_ready_row(self) -> None:
        profile = HardwareProfile.linear(
            target_overhead_ms=10.0,
            target_per_row_ms=0.1,
            verifier_slot_ms=0.0,
        )
        policy = FissionSpecPolicy(max_wait_ms=1.0)
        # A newer arrival at t=.9 cannot reset the row that has waited since 0.
        ctx = context(
            profile,
            now=0.9,
            oldest=0.0,
            next_time=1.1,
            next_count=1,
        )
        self.assertEqual(policy.dispatch_at(ctx), 0.9)

    def test_infeasible_fusion_dispatches_even_when_its_cost_is_lower(self) -> None:
        profile = HardwareProfile.linear(
            target_overhead_ms=1.0,
            target_per_row_ms=0.0 + 0.1,
            verifier_slot_ms=0.0,
        )
        policy = FissionSpecPolicy(max_wait_ms=1.0)
        ctx = context(
            profile,
            next_time=0.1,
            deadline=1.15,
            future_deadline=100.0,
        )
        self.assertEqual(policy.dispatch_at(ctx), 0.0)

    def test_feasible_fusion_overrides_infeasible_launch_plan(self) -> None:
        profile = HardwareProfile.linear(
            target_overhead_ms=10.0,
            target_per_row_ms=0.1,
            verifier_slot_ms=0.0,
        )
        policy = FissionSpecPolicy(max_wait_ms=1.0)
        ctx = context(
            profile,
            next_time=0.1,
            deadline=100.0,
            future_deadline=10.5,
        )
        self.assertEqual(policy.dispatch_at(ctx), 0.1)
        self.assertEqual(
            policy.dispatch_at(context(profile, next_time=0.1, deadline=1.0)),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
