from __future__ import annotations

import unittest

from fissionspec.oracle import (
    OracleAction,
    OracleLimit,
    OracleLimitExceeded,
    aggregate_request_flow_time_ms,
    offline_coalescing_oracle,
)
from fissionspec.profiles import HardwareProfile, LatencyCurve
from fissionspec.rng import CounterRNG
from fissionspec.workload import RequestConfig, Workload


def oracle_profile() -> HardwareProfile:
    return HardwareProfile(
        target_curve=LatencyCurve(((1, 1.0), (2, 1.25), (4, 1.5))),
        draft_curve=LatencyCurve(((1, 0.1), (2, 0.15), (4, 0.2))),
        recovery_curve=LatencyCurve(((1, 0.2), (2, 0.3), (4, 0.4))),
        verifier_slot_ms=0.0,
        name="oracle-test",
    )


def one_token_workload(second_arrival_ms: float, *, deadline_ms: float = 50.0) -> Workload:
    return Workload(
        (
            RequestConfig(
                request_id="first",
                arrival_ms=0.0,
                output_tokens=1,
                speculation_length=1,
                cache_hit_probability=1.0,
                token_acceptance_probability=1.0,
                deadline_ms=deadline_ms,
            ),
            RequestConfig(
                request_id="second",
                arrival_ms=second_arrival_ms,
                output_tokens=1,
                speculation_length=1,
                cache_hit_probability=1.0,
                token_acceptance_probability=1.0,
            ),
        ),
        name="two-one-token-requests",
    )


class OfflineOracleTests(unittest.TestCase):
    def solve(self, workload: Workload, *, max_simulations: int = 16):
        return offline_coalescing_oracle(
            workload,
            oracle_profile(),
            CounterRNG("oracle-tests"),
            max_batch_size=2,
            max_decision_depth=4,
            max_simulations=max_simulations,
        )

    def test_batching_benefit_trace_chooses_wait(self) -> None:
        solved = self.solve(one_token_workload(0.2))

        self.assertEqual(solved.action_prefix, (OracleAction.WAIT_NEXT_READINESS,))
        self.assertEqual(
            solved.best_result.target_launches[0].request_ids,
            ("first", "second"),
        )
        self.assertAlmostEqual(solved.objective_flow_time_ms, 2.7)

    def test_late_readiness_and_tight_deadline_trace_chooses_dispatch(self) -> None:
        solved = self.solve(one_token_workload(0.9, deadline_ms=1.05))

        self.assertEqual(solved.action_prefix, (OracleAction.DISPATCH_NOW,))
        self.assertEqual(solved.best_result.target_launches[0].request_ids, ("first",))
        first = next(
            request for request in solved.best_result.requests if request.request_id == "first"
        )
        self.assertLessEqual(first.completion_ms, 1.05)

    def test_exact_tie_prefers_dispatch_deterministically(self) -> None:
        # With L(1)=1, L(2)=1.25, and delta=.25, both paths have total
        # flow time 2.75 ms.  The specified tie order chooses dispatch.
        solved = self.solve(one_token_workload(0.25))

        self.assertEqual(solved.objective_flow_time_ms, 2.75)
        self.assertEqual(solved.action_prefix, (OracleAction.DISPATCH_NOW,))

    def test_repeated_search_is_deterministic_and_result_is_consistent(self) -> None:
        workload = one_token_workload(0.2)
        first = self.solve(workload)
        second = self.solve(workload)

        self.assertEqual(first, second)
        self.assertEqual(first.nodes, first.simulations)
        self.assertEqual(first.nodes, 3)
        self.assertEqual(first.leaves, 2)
        self.assertEqual(first.max_depth, 1)
        self.assertEqual(
            first.objective_flow_time_ms,
            aggregate_request_flow_time_ms(first.best_result),
        )

    def test_decision_depth_limit_fails_instead_of_approximating(self) -> None:
        with self.assertRaises(OracleLimitExceeded) as raised:
            offline_coalescing_oracle(
                one_token_workload(0.2),
                oracle_profile(),
                CounterRNG(1),
                max_batch_size=2,
                max_decision_depth=0,
                max_simulations=16,
            )

        self.assertEqual(raised.exception.kind, OracleLimit.DECISION_DEPTH)
        self.assertEqual(raised.exception.limit, 0)
        self.assertEqual(raised.exception.observed, 1)
        self.assertEqual(raised.exception.action_prefix, ())

    def test_simulation_limit_fails_instead_of_returning_best_so_far(self) -> None:
        with self.assertRaises(OracleLimitExceeded) as raised:
            self.solve(one_token_workload(0.2), max_simulations=1)

        self.assertEqual(raised.exception.kind, OracleLimit.SIMULATIONS)
        self.assertEqual(raised.exception.limit, 1)
        self.assertEqual(raised.exception.observed, 2)

    def test_limit_types_reject_bool_and_invalid_values(self) -> None:
        workload = one_token_workload(0.2)
        with self.assertRaisesRegex(ValueError, "max_decision_depth"):
            offline_coalescing_oracle(
                workload,
                oracle_profile(),
                CounterRNG(1),
                max_decision_depth=True,  # type: ignore[arg-type]
                max_simulations=4,
            )
        with self.assertRaisesRegex(ValueError, "max_simulations"):
            offline_coalescing_oracle(
                workload,
                oracle_profile(),
                CounterRNG(1),
                max_decision_depth=2,
                max_simulations=0,
            )


if __name__ == "__main__":
    unittest.main()
